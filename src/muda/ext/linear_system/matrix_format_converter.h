#pragma once
#include <muda/ext/linear_system/device_dense_matrix.h>
#include <muda/ext/linear_system/device_dense_vector.h>
#include <muda/ext/linear_system/device_triplet_matrix.h>
#include <muda/ext/linear_system/device_doublet_vector.h>
#include <muda/ext/linear_system/device_bcoo_matrix.h>
#include <muda/ext/linear_system/device_bcoo_vector.h>
#include <muda/ext/linear_system/device_bsr_matrix.h>
#include <muda/ext/linear_system/device_csr_matrix.h>
namespace muda
{
namespace details
{
    template <typename T, int N>
    class MatrixFormatConverter
    {
        using BlockMatrix   = DeviceTripletMatrix<T, N>::BlockMatrix;
        using SegmentVector = DeviceDoubletVector<T, N>::SegmentVector;

        muda::DeviceBuffer<int> sort_index;
        muda::DeviceBuffer<int> sort_index_tmp;

        muda::DeviceBuffer<std::byte> workspace;
        muda::DeviceBuffer<int>       col_tmp;
        muda::DeviceBuffer<int>       row_tmp;

        muda::DeviceBuffer<int> unique_indices;
        muda::DeviceBuffer<int> unique_counts;
        muda::DeviceBuffer<int> offsets;

        muda::DeviceVar<int> count;

        muda::DeviceBuffer<int2> ij_pairs;
        muda::DeviceBuffer<int2> unique_ij_pairs;

        muda::DeviceBuffer<BlockMatrix>   unique_blocks;
        muda::DeviceBuffer<SegmentVector> unique_segments;
        muda::DeviceBuffer<SegmentVector> temp_segments;

        muda::DeviceBuffer<T> unique_values;

      public:
        // Triplet -> BCOO
        void convert(const DeviceTripletMatrix<T, N>& from, DeviceBCOOMatrix<T, N>& to);

        void merge_sort_indices_and_blocks(const DeviceTripletMatrix<T, N>& from,
                                           DeviceBCOOMatrix<T, N>& to);
        void make_unique_indices(const DeviceTripletMatrix<T, N>& from,
                                 DeviceBCOOMatrix<T, N>&          to);
        void make_unique_blocks(const DeviceTripletMatrix<T, N>& from,
                                DeviceBCOOMatrix<T, N>&          to);


        // BCOO -> Dense Matrix
        void convert(const DeviceBCOOMatrix<T, N>& from,
                     DeviceDenseMatrix<T>&         to,
                     bool                          clear_dense_matrix = true);

        // BCOO -> COO
        void convert(const DeviceBCOOMatrix<T, N>& from, DeviceCOOMatrix<T>& to);
        void expand_blocks(const DeviceBCOOMatrix<T, N>& from, DeviceCOOMatrix<T>& to);
        void sort_indices_and_values(const DeviceBCOOMatrix<T, N>& from,
                                     DeviceCOOMatrix<T>&           to);

        // BCOO -> BSR
        void convert(const DeviceBCOOMatrix<T, N>& from, DeviceBSRMatrix<T, N>& to);
        void convert(DeviceBCOOMatrix<T, N>&& from, DeviceBSRMatrix<T, N>& to);
        void calculate_block_offsets(const DeviceBCOOMatrix<T, N>& from,
                                     DeviceBSRMatrix<T, N>&        to);


        // Doublet -> Dense Vector
        void convert(const DeviceDoubletVector<T, N>& from,
                     DeviceDenseVector<T>&            to,
                     bool clear_dense_vector = true);
        void merge_sort_indices_and_segments(const DeviceDoubletVector<T, N>& from,
                                             DeviceDenseVector<T>& to);
        void make_unique_indices(const DeviceDoubletVector<T, N>& from,
                                 DeviceDenseVector<T>&            to);
        void make_unique_segments(const DeviceDoubletVector<T, N>& from,
                                  DeviceDenseVector<T>&            to);
        void set_unique_segments_to_dense_vector(const DeviceDoubletVector<T, N>& from,
                                                 DeviceDenseVector<T>& to,
                                                 bool clear_dense_vector);

        // BSR -> CSR
        void convert(const DeviceBSRMatrix<T, N>& from, DeviceCSRMatrix<T>& to);
    };

    template <typename T>
    class MatrixFormatConverter<T, 1>
    {
        muda::DeviceBuffer<int> sort_index;
        muda::DeviceBuffer<int> sort_index_tmp;

        muda::DeviceBuffer<std::byte> workspace;
        muda::DeviceBuffer<int>       col_tmp;
        muda::DeviceBuffer<int>       row_tmp;

        muda::DeviceBuffer<int> unique_indices;
        muda::DeviceBuffer<int> unique_counts;
        muda::DeviceBuffer<int> offsets;

        muda::DeviceVar<int> count;

        muda::DeviceBuffer<int2> ij_pairs;
        muda::DeviceBuffer<int2> unique_ij_pairs;

        muda::DeviceBuffer<T> unique_values;
        muda::DeviceBuffer<T> temp_values;

      public:
        // Triplet -> COO
        void convert(const DeviceTripletMatrix<T, 1>& from, DeviceCOOMatrix<T>& to);

        void merge_sort_indices_and_values(const DeviceTripletMatrix<T, 1>& from,
                                           DeviceBCOOMatrix<T, 1>& to);
        void make_unique_indices(const DeviceTripletMatrix<T, 1>& from,
                                 DeviceCOOMatrix<T>&              to);
        void make_unique_values(const DeviceTripletMatrix<T, 1>& from,
                                DeviceCOOMatrix<T>&              to);


        // COO -> Dense Matrix
        void convert(const DeviceBCOOMatrix<T, 1>& from,
                     DeviceDenseMatrix<T>&         to,
                     bool                          clear_dense_matrix = true);

        // COO -> CSR
        void convert(const DeviceBCOOMatrix<T, 1>& from, DeviceBSRMatrix<T, 1>& to);
        void convert(DeviceBCOOMatrix<T, 1>&& from, DeviceBSRMatrix<T, 1>& to);
        void calculate_block_offsets(const DeviceBCOOMatrix<T, 1>& from,
                                     DeviceBSRMatrix<T, 1>&        to);


        // Doublet -> Dense Vector
        void convert(const DeviceDoubletVector<T, 1>& from,
                     DeviceDenseVector<T>&            to,
                     bool clear_dense_vector = true);
        void merge_sort_indices_and_segments(const DeviceDoubletVector<T, 1>& from,
                                             DeviceDenseVector<T>& to);
        void make_unique_indices(const DeviceDoubletVector<T, 1>& from,
                                 DeviceDenseVector<T>&            to);
        void make_unique_segments(const DeviceDoubletVector<T, 1>& from,
                                  DeviceDenseVector<T>&            to);
        void set_unique_segments_to_dense_vector(const DeviceDoubletVector<T, 1>& from,
                                                 DeviceDenseVector<T>& to,
                                                 bool clear_dense_vector);
    };
}  // namespace details
}  // namespace muda


namespace muda
{
template <typename T, int N>
class MatrixFormatConverter
{
    details::MatrixFormatConverter<T, N> impl;
};
}  // namespace muda

#include "details/matrix_format_converter_block.inl"
#include "details/matrix_format_converter.inl"
