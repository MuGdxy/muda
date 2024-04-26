#include <muda/cub/device/device_merge_sort.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <muda/cub/device/device_scan.h>
#include <muda/cub/device/device_segmented_reduce.h>
#include <muda/launch.h>
#include <muda/profiler.h>
// for encode run length usage
MUDA_GENERIC constexpr bool operator==(const int2& a, const int2& b)
{
    return a.x == b.x && a.y == b.y;
}

namespace muda::details
{
//using T         = float;
//constexpr int N = 3;

template <typename T, int N>
void MatrixFormatConverter<T, N>::convert(const DeviceTripletMatrix<T, N>& from,
                                          DeviceBCOOMatrix<T, N>&          to)
{
    to.reshape(from.block_rows(), from.block_cols());
    to.resize_triplets(from.triplet_count());


    if(to.triplet_count() == 0)
        return;

    merge_sort_indices_and_blocks(from, to);
    make_unique_indices(from, to);
    make_unique_blocks(from, to);
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::merge_sort_indices_and_blocks(
    const DeviceTripletMatrix<T, N>& from, DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto src_row_indices = from.block_row_indices();
    auto src_col_indices = from.block_col_indices();
    auto src_blocks      = from.block_values();

    loose_resize(sort_index, src_row_indices.size());
    loose_resize(ij_pairs, src_row_indices.size());

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(src_row_indices.size(),
               [row_indices = src_row_indices.cviewer().name("row_indices"),
                col_indices = src_col_indices.cviewer().name("col_indices"),
                ij_pairs = ij_pairs.viewer().name("ij_pairs")] __device__(int i) mutable
               {
                   ij_pairs(i).x = row_indices(i);
                   ij_pairs(i).y = col_indices(i);
               });

    ParallelFor(256)
        .kernel_name(__FUNCTION__)  //
        .apply(src_row_indices.size(),
               [sort_index = sort_index.viewer().name("sort_index")] __device__(int i) mutable
               { sort_index(i) = i; });

    DeviceMergeSort().SortPairs(ij_pairs.data(),
                                sort_index.data(),
                                ij_pairs.size(),
                                [] __device__(const int2& a, const int2& b) {
                                    return a.x < b.x || (a.x == b.x && a.y < b.y);
                                });


    // set ij_pairs back to row_indices and col_indices

    auto dst_row_indices = to.block_row_indices();
    auto dst_col_indices = to.block_col_indices();

    ParallelFor(256)
        .kernel_name("set col row indices")
        .apply(dst_row_indices.size(),
               [row_indices = dst_row_indices.viewer().name("row_indices"),
                col_indices = dst_col_indices.viewer().name("col_indices"),
                ij_pairs = ij_pairs.viewer().name("ij_pairs")] __device__(int i) mutable
               {
                   row_indices(i) = ij_pairs(i).x;
                   col_indices(i) = ij_pairs(i).y;
               });


    // sort the block values

    loose_resize(unique_blocks, from.m_block_values.size());

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(src_blocks.size(),
               [src_blocks = src_blocks.cviewer().name("blocks"),
                sort_index = sort_index.cviewer().name("sort_index"),
                dst_blocks = unique_blocks.viewer().name("block_values")] __device__(int i) mutable
               { dst_blocks(i) = src_blocks(sort_index(i)); });
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::make_unique_indices(const DeviceTripletMatrix<T, N>& from,
                                                      DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto& row_indices = to.m_block_row_indices;
    auto& col_indices = to.m_block_col_indices;

    loose_resize(unique_ij_pairs, ij_pairs.size());
    loose_resize(unique_counts, ij_pairs.size());


    DeviceRunLengthEncode().Encode(ij_pairs.data(),
                                   unique_ij_pairs.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   ij_pairs.size());

    int h_count = count;

    unique_ij_pairs.resize(h_count);
    unique_counts.resize(h_count);

    offsets.resize(unique_counts.size() + 1);  // +1 for the last offset_end

    DeviceScan().ExclusiveSum(
        unique_counts.data(), offsets.data(), unique_counts.size());


    muda::ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(unique_counts.size(),
               [unique_ij_pairs = unique_ij_pairs.viewer().name("unique_ij_pairs"),
                row_indices = row_indices.viewer().name("row_indices"),
                col_indices = col_indices.viewer().name("col_indices")] __device__(int i) mutable
               {
                   row_indices(i) = unique_ij_pairs(i).x;
                   col_indices(i) = unique_ij_pairs(i).y;
               });

    row_indices.resize(h_count);
    col_indices.resize(h_count);
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::make_unique_blocks(const DeviceTripletMatrix<T, N>& from,
                                                     DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto& row_indices = to.m_block_row_indices;
    auto& blocks      = to.m_block_values;
    blocks.resize(row_indices.size());
    // first we add the offsets to counts, to get the offset_ends

    Launch()
        .kernel_name(__FUNCTION__)
        .apply([offsets = offsets.viewer().name("offset"),
                counts  = unique_counts.cviewer().name("counts"),
                last    = unique_counts.size() - 1] __device__() mutable
               { offsets(last + 1) = offsets(last) + counts(last); });

    auto& begin_offset = offsets;
    auto& end_offset   = unique_counts;  // already contains the offset_ends

    // then we do a segmented reduce to get the unique blocks

    DeviceSegmentedReduce().Reduce(
        unique_blocks.data(),
        blocks.data(),
        blocks.size(),
        offsets.data(),
        offsets.data() + 1,
        [] __host__ __device__(const BlockMatrix& a, const BlockMatrix& b) -> BlockMatrix
        { return a + b; },
        BlockMatrix::Zero().eval());
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::convert(const DeviceBCOOMatrix<T, N>& from,
                                          DeviceDenseMatrix<T>&         to,
                                          bool clear_dense_matrix)
{
    using namespace muda;
    auto size = N * from.block_rows();
    to.reshape(size, size);

    if(clear_dense_matrix)
        to.fill(0);

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(from.block_values().size(),
               [blocks = from.cviewer().name("src_sparse_matrix"),
                dst = to.viewer().name("dst_dense_matrix")] __device__(int i) mutable
               {
                   auto block = blocks(i);
                   auto row   = block.block_row_index * N;
                   auto col   = block.block_col_index * N;
                   dst.block<N, N>(row, col).as_eigen() += block.block_value;
               });
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::convert(const DeviceBCOOMatrix<T, N>& from,
                                          DeviceBSRMatrix<T, N>&        to)
{
    calculate_block_offsets(from, to);

    to.m_block_col_indices = from.m_block_col_indices;
    to.m_block_values      = from.m_block_values;
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::convert(DeviceBCOOMatrix<T, N>&& from,
                                          DeviceBSRMatrix<T, N>&   to)
{
    calculate_block_offsets(from, to);
    to.m_block_col_indices = std::move(from.m_block_col_indices);
    to.m_block_values      = std::move(from.m_block_values);
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::calculate_block_offsets(const DeviceBCOOMatrix<T, N>& from,
                                                          DeviceBSRMatrix<T, N>& to)
{
    using namespace muda;
    to.reshape(from.block_rows(), from.block_cols());

    auto& dst_row_offsets = to.m_block_row_offsets;

    // alias the offsets to the col_counts_per_row(reuse)
    auto& col_counts_per_row = offsets;
    col_counts_per_row.resize(to.m_block_row_offsets.size());
    col_counts_per_row.fill(0);

    unique_indices.resize(from.non_zero_blocks());
    unique_counts.resize(from.non_zero_blocks());

    // run length encode the row
    DeviceRunLengthEncode().Encode(from.m_block_row_indices.data(),
                                   unique_indices.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   from.non_zero_blocks());
    int h_count = count;

    unique_indices.resize(h_count);
    unique_counts.resize(h_count);

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(unique_counts.size(),
               [unique_indices     = unique_indices.cviewer().name("offset"),
                counts             = unique_counts.viewer().name("counts"),
                col_counts_per_row = col_counts_per_row.viewer().name(
                    "col_counts_per_row")] __device__(int i) mutable
               {
                   auto row                = unique_indices(i);
                   col_counts_per_row(row) = counts(i);
               });

    // calculate the offsets
    DeviceScan().ExclusiveSum(col_counts_per_row.data(),
                              dst_row_offsets.data(),
                              col_counts_per_row.size());
}
template <typename T, int N>
void MatrixFormatConverter<T, N>::convert(const DeviceBCOOVector<T, N>& from,
                                          DeviceDenseVector<T>&         to,
                                          bool clear_dense_vector)
{
    to.resize(N * from.segment_count());
    set_unique_segments_to_dense_vector(from, to, clear_dense_vector);
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::convert(const DeviceDoubletVector<T, N>& from,
                                          DeviceBCOOVector<T, N>&          to)
{
    to.reshape(from.segment_count());
    to.resize_doublet(N * from.doublet_count());
    merge_sort_indices_and_segments(from, to);
    make_unique_indices(from, to);
    make_unique_segments(from, to);
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::merge_sort_indices_and_segments(
    const DeviceDoubletVector<T, N>& from, DeviceBCOOVector<T, N>& to)
{
    using namespace muda;

    auto& indices = sort_index;  // alias sort_index to index

    // copy as temp
    indices       = from.m_segment_indices;
    temp_segments = from.m_segment_values;

    DeviceMergeSort().SortPairs(indices.data(),
                                temp_segments.data(),
                                indices.size(),
                                [] __device__(const int& a, const int& b)
                                { return a < b; });
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::make_unique_indices(const DeviceDoubletVector<T, N>& from,
                                                      DeviceBCOOVector<T, N>& to)
{
    using namespace muda;

    auto& indices        = sort_index;  // alias sort_index to index
    auto& unique_indices = to.m_segment_indices;

    loose_resize(unique_indices, indices.size());
    loose_resize(unique_counts, indices.size());

    DeviceRunLengthEncode().Encode(indices.data(),
                                   unique_indices.data(),
                                   unique_counts.data(),
                                   count.data(),
                                   indices.size());

    int h_count = count;

    unique_indices.resize(h_count);
    unique_counts.resize(h_count);

    loose_resize(offsets, unique_counts.size() + 1);

    DeviceScan().ExclusiveSum(
        unique_counts.data(), offsets.data(), unique_counts.size());

    // calculate the offset_ends, and set to the unique_counts

    auto& begin_offset = offsets;

    Launch()
        .kernel_name(__FUNCTION__)
        .apply([offset = offsets.viewer().name("offset"),
                count  = unique_counts.cviewer().name("counts"),
                last   = unique_counts.size() - 1] __device__() mutable
               { offset(last + 1) = offset(last) + count(last); });
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::make_unique_segments(const DeviceDoubletVector<T, N>& from,
                                                       DeviceBCOOVector<T, N>& to)
{
    using namespace muda;

    auto& begin_offset = offsets;
    auto& end_offset   = unique_counts;

    auto& unique_indices  = to.m_segment_indices;
    auto& unique_segments = to.m_segment_values;

    unique_segments.resize(unique_indices.size());

    DeviceSegmentedReduce().Reduce(
        temp_segments.data(),
        unique_segments.data(),
        unique_segments.size(),
        begin_offset.data(),
        begin_offset.data() + 1,
        [] __host__ __device__(const SegmentVector& a, const SegmentVector& b) -> SegmentVector
        { return a + b; },
        SegmentVector::Zero().eval());
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::set_unique_segments_to_dense_vector(
    const DeviceBCOOVector<T, N>& from, DeviceDenseVector<T>& to, bool clear_dense_vector)
{
    using namespace muda;

    if(clear_dense_vector)
        to.fill(0);

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(from.non_zero_segments(),
               [unique_segments = from.m_segment_values.cviewer().name("unique_segments"),
                unique_indices = from.m_segment_indices.cviewer().name("unique_indices"),
                dst = to.viewer().name("dst_dense_vector")] __device__(int i) mutable
               {
                   auto index = unique_indices(i);
                   dst.segment<N>(index * N).as_eigen() += unique_segments(i);
               });
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::convert(const DeviceDoubletVector<T, N>& from,
                                          DeviceDenseVector<T>&            to,
                                          bool clear_dense_vector)
{
    using namespace muda;

    to.resize(N * from.segment_count());

    if(clear_dense_vector)
        to.fill(0);

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(from.doublet_count(),
               [src = from.viewer().name("src_sparse_vector"),
                dst = to.viewer().name("dst_dense_vector")] __device__(int i) mutable
               {
                   auto&& [index, value] = src(i);
                   dst.segment<N>(index * N).atomic_add(value);
               });
}

template <typename T>
void bsr2csr(cusparseHandle_t         handle,
             int                      mb,
             int                      nb,
             int                      blockDim,
             cusparseMatDescr_t       descrA,
             const T*                 bsrValA,
             const int*               bsrRowPtrA,
             const int*               bsrColIndA,
             int                      nnzb,
             DeviceCSRMatrix<T>&      to,
             muda::DeviceBuffer<int>& row_offsets,
             muda::DeviceBuffer<int>& col_indices,
             muda::DeviceBuffer<T>&   values)
{
    using namespace muda;
    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
    int                 m   = mb * blockDim;
    int                 nnz = nnzb * blockDim * blockDim;  // number of elements
    to.reshape(m, m);
    col_indices.resize(nnz);
    values.resize(nnz);
    if constexpr(std::is_same_v<T, float>)
    {
        checkCudaErrors(cusparseSbsr2csr(handle,
                                         dir,
                                         mb,
                                         nb,
                                         descrA,
                                         bsrValA,
                                         bsrRowPtrA,
                                         bsrColIndA,
                                         blockDim,
                                         to.legacy_descr(),
                                         values.data(),
                                         row_offsets.data(),
                                         col_indices.data()));
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        checkCudaErrors(cusparseDbsr2csr(handle,
                                         dir,
                                         mb,
                                         nb,
                                         descrA,
                                         bsrValA,
                                         bsrRowPtrA,
                                         bsrColIndA,
                                         blockDim,
                                         to.legacy_descr(),
                                         values.data(),
                                         row_offsets.data(),
                                         col_indices.data()));
    }
}


template <typename T, int N>
void MatrixFormatConverter<T, N>::convert(const DeviceBCOOMatrix<T, N>& from,
                                          DeviceCOOMatrix<T>&           to)
{
    expand_blocks(from, to);
    sort_indices_and_values(from, to);
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::expand_blocks(const DeviceBCOOMatrix<T, N>& from,
                                                DeviceCOOMatrix<T>& to)
{
    using namespace muda;

    constexpr int N2 = N * N;

    to.reshape(from.block_rows() * N, from.block_cols() * N);
    to.resize_triplets(from.non_zero_blocks() * N2);

    auto& row_indices = to.m_row_indices;
    auto& col_indices = to.m_col_indices;
    auto& values      = to.m_values;

    auto& block_row_indices = from.m_block_row_indices;
    auto& block_col_indices = from.m_block_col_indices;
    auto& block_values      = from.m_block_values;


    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(block_row_indices.size(),
               [block_row_indices = block_row_indices.cviewer().name("block_row_indices"),
                block_col_indices = block_col_indices.cviewer().name("block_col_indices"),
                block_values = block_values.cviewer().name("block_values"),
                row_indices  = row_indices.viewer().name("row_indices"),
                col_indices  = col_indices.viewer().name("col_indices"),
                values = values.viewer().name("values")] __device__(int i) mutable
               {
                   auto block_row_index = block_row_indices(i);
                   auto block_col_index = block_col_indices(i);
                   auto block           = block_values(i);

                   auto row = block_row_index * N;
                   auto col = block_col_index * N;

                   auto index = i * N2;
#pragma unroll
                   for(int r = 0; r < N; ++r)
                   {
#pragma unroll
                       for(int c = 0; c < N; ++c)
                       {
                           row_indices(index) = row + r;
                           col_indices(index) = col + c;
                           values(index)      = block(r, c);
                           ++index;
                       }
                   }
               });
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::sort_indices_and_values(const DeviceBCOOMatrix<T, N>& from,
                                                          DeviceCOOMatrix<T>& to)
{
    using namespace muda;

    auto& row_indices = to.m_row_indices;
    auto& col_indices = to.m_col_indices;
    auto& values      = to.m_values;

    ij_pairs.resize(row_indices.size());

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(row_indices.size(),
               [row_indices = row_indices.cviewer().name("row_indices"),
                col_indices = col_indices.cviewer().name("col_indices"),
                ij_pairs = ij_pairs.viewer().name("ij_pairs")] __device__(int i) mutable
               {
                   ij_pairs(i).x = row_indices(i);
                   ij_pairs(i).y = col_indices(i);
               });

    DeviceMergeSort().SortPairs(ij_pairs.data(),
                                to.m_values.data(),
                                ij_pairs.size(),
                                [] __device__(const int2& a, const int2& b) {
                                    return a.x < b.x || (a.x == b.x && a.y < b.y);
                                });

    // set ij_pairs back to row_indices and col_indices

    auto dst_row_indices = to.row_indices();
    auto dst_col_indices = to.col_indices();

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(dst_row_indices.size(),
               [row_indices = dst_row_indices.viewer().name("row_indices"),
                col_indices = dst_col_indices.viewer().name("col_indices"),
                ij_pairs = ij_pairs.viewer().name("ij_pairs")] __device__(int i) mutable
               {
                   row_indices(i) = ij_pairs(i).x;
                   col_indices(i) = ij_pairs(i).y;
               });
}

template <typename T, int N>
void MatrixFormatConverter<T, N>::convert(const DeviceBSRMatrix<T, N>& from,
                                          DeviceCSRMatrix<T>&          to)
{
    using namespace muda;

    bsr2csr(cusparse(),
            from.block_rows(),
            from.block_cols(),
            N,
            from.legacy_descr(),
            (const T*)from.m_block_values.data(),
            from.m_block_row_offsets.data(),
            from.m_block_col_indices.data(),
            from.non_zero_blocks(),
            to,
            to.m_row_offsets,
            to.m_col_indices,
            to.m_values);
}
}  // namespace muda::details