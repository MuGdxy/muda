#pragma once
#include <muda/ext/linear_system/linear_system_handles.h>
#include <muda/ext/linear_system/device_dense_matrix.h>
#include <muda/ext/linear_system/device_dense_vector.h>
#include <muda/ext/linear_system/device_triplet_matrix.h>
#include <muda/ext/linear_system/device_doublet_vector.h>
#include <muda/ext/linear_system/device_bcoo_matrix.h>
#include <muda/ext/linear_system/device_bcoo_vector.h>
#include <muda/ext/linear_system/device_bsr_matrix.h>
#include <muda/ext/linear_system/device_csr_matrix.h>
#include <muda/type_traits/cuda_arch.h>

namespace muda
{
namespace details
{
    class MatrixFormatConverterBase
    {
      protected:
        LinearSystemHandles& m_handles;
        cudaDataType_t       m_data_type;
        int                  m_N;

      public:
        MatrixFormatConverterBase(LinearSystemHandles& context, cudaDataType_t data_type, int N)
            : m_handles(context)
            , m_data_type(data_type)
            , m_N(N)
        {
        }

        virtual ~MatrixFormatConverterBase() = default;

        auto dim() const { return m_N; }
        auto data_type() const { return m_data_type; }
        auto cublas() const { return m_handles.cublas(); }
        auto cusparse() const { return m_handles.cusparse(); }
        auto cusolver_sp() const { return m_handles.cusolver_sp(); }
        auto cusolver_dn() const { return m_handles.cusolver_dn(); }

        template <typename T>
        void loose_resize(muda::DeviceBuffer<T>& buf, size_t new_size)
        {
            if(buf.capacity() < new_size)
                buf.reserve(new_size * m_handles.reserve_ratio());
            buf.resize(new_size);
        }
    };

    template <typename T, int N>
    class MatrixFormatConverter : public MatrixFormatConverterBase
    {
        using BlockMatrix   = typename DeviceTripletMatrix<T, N>::BlockMatrix;
        using SegmentVector = typename DeviceDoubletVector<T, N>::SegmentVector;

        muda::DeviceBuffer<int> sort_index;
        muda::DeviceBuffer<int> sort_index_tmp;

        muda::DeviceBuffer<int> col_tmp;
        muda::DeviceBuffer<int> row_tmp;

        muda::DeviceBCOOMatrix<T, N> temp_bcoo_matrix;
        muda::DeviceBCOOVector<T, N> temp_bcoo_vector;

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
        MatrixFormatConverter(LinearSystemHandles& handles)
            : MatrixFormatConverterBase(handles, cuda_data_type<T>(), N)
        {
        }

        virtual ~MatrixFormatConverter() = default;


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

        // Doublet -> BCOO
        void convert(const DeviceDoubletVector<T, N>& from, DeviceBCOOVector<T, N>& to);

        void merge_sort_indices_and_segments(const DeviceDoubletVector<T, N>& from,
                                             DeviceBCOOVector<T, N>& to);
        void make_unique_indices(const DeviceDoubletVector<T, N>& from,
                                 DeviceBCOOVector<T, N>&          to);
        void make_unique_segments(const DeviceDoubletVector<T, N>& from,
                                  DeviceBCOOVector<T, N>&          to);

        // BCOO -> Dense Vector
        void convert(const DeviceBCOOVector<T, N>& from,
                     DeviceDenseVector<T>&         to,
                     bool                          clear_dense_vector = true);
        void set_unique_segments_to_dense_vector(const DeviceBCOOVector<T, N>& from,
                                                 DeviceDenseVector<T>& to,
                                                 bool clear_dense_vector);

        // Triplet -> Dense Vector
        void convert(const DeviceDoubletVector<T, N>& from,
                     DeviceDenseVector<T>&            to,
                     bool clear_dense_vector = true);

        // BSR -> CSR
        void convert(const DeviceBSRMatrix<T, N>& from, DeviceCSRMatrix<T>& to);
    };

    template <typename T>
    class MatrixFormatConverter<T, 1> : public MatrixFormatConverterBase
    {
        muda::DeviceBuffer<int> sort_index;
        muda::DeviceBuffer<int> sort_index_tmp;

        muda::DeviceBuffer<int> col_tmp;
        muda::DeviceBuffer<int> row_tmp;

        muda::DeviceBuffer<int>  unique_indices;
        muda::DeviceCOOMatrix<T> temp_coo_matrix;
        muda::DeviceCOOVector<T> temp_coo_vector;

        muda::DeviceBuffer<int> unique_counts;
        muda::DeviceBuffer<int> offsets;

        muda::DeviceVar<int> count;

        muda::DeviceBuffer<int2> ij_pairs;
        muda::DeviceBuffer<int2> unique_ij_pairs;

        muda::DeviceBuffer<T> unique_values;
        muda::DeviceBuffer<T> temp_values;

      public:
        MatrixFormatConverter(LinearSystemHandles& handles)
            : MatrixFormatConverterBase(handles, cuda_data_type<T>(), 1)
        {
        }

        virtual ~MatrixFormatConverter() = default;

        // Triplet -> COO
        void convert(const DeviceTripletMatrix<T, 1>& from, DeviceCOOMatrix<T>& to);
        void merge_sort_indices_and_values(const DeviceTripletMatrix<T, 1>& from,
                                           DeviceCOOMatrix<T>& to);
        void make_unique_indices(const DeviceTripletMatrix<T, 1>& from,
                                 DeviceCOOMatrix<T>&              to);
        void make_unique_values(const DeviceTripletMatrix<T, 1>& from,
                                DeviceCOOMatrix<T>&              to);


        // COO -> Dense Matrix
        void convert(const DeviceCOOMatrix<T>& from,
                     DeviceDenseMatrix<T>&     to,
                     bool                      clear_dense_matrix = true);

        // COO -> CSR
        void convert(const DeviceCOOMatrix<T>& from, DeviceCSRMatrix<T>& to);
        void convert(DeviceCOOMatrix<T>&& from, DeviceCSRMatrix<T>& to);
        void calculate_block_offsets(const DeviceCOOMatrix<T>& from,
                                     DeviceCSRMatrix<T>&       to);

        // Doublet -> COO
        void convert(const DeviceDoubletVector<T, 1>& from, DeviceCOOVector<T>& to);
        void merge_sort_indices_and_values(const DeviceDoubletVector<T, 1>& from,
                                           DeviceCOOVector<T>& to);
        void make_unique_indices(const DeviceDoubletVector<T, 1>& from,
                                 DeviceCOOVector<T>&              to);
        void make_unique_values(const DeviceDoubletVector<T, 1>& from,
                                DeviceCOOVector<T>&              to);


        // COO -> Dense Vector
        void convert(const DeviceCOOVector<T>& from,
                     DeviceDenseVector<T>&     to,
                     bool                      clear_dense_vector = true);
        void set_unique_values_to_dense_vector(const DeviceDoubletVector<T, 1>& from,
                                               DeviceDenseVector<T>& to,
                                               bool clear_dense_vector);

        // Triplet -> Dense Vector
        void convert(const DeviceDoubletVector<T, 1>& from,
                     DeviceDenseVector<T>&            to,
                     bool clear_dense_vector = true);
    };
}  // namespace details
}  // namespace muda

#include "details/matrix_format_converter_impl_block.inl"
#include "details/matrix_format_converter_impl.inl"
