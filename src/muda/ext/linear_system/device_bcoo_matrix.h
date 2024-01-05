#pragma once
#include <muda/buffer/device_buffer.h>
#include <muda/ext/linear_system/bcoo_matrix_viewer.h>
#include <muda/ext/linear_system/device_triplet_matrix.h>

namespace muda
{
template <typename T, int N>
class DeviceBCOOMatrix : public DeviceTripletMatrix<T, N>
{
    friend class details::MatrixFormatConverter<T, N>;

  public:
    using BlockMatrix = Eigen::Matrix<T, N, N>;

    DeviceBCOOMatrix()                                         = default;
    ~DeviceBCOOMatrix()                                        = default;
    DeviceBCOOMatrix(const DeviceBCOOMatrix& other)            = default;
    DeviceBCOOMatrix(DeviceBCOOMatrix&& other)                 = default;
    DeviceBCOOMatrix& operator=(const DeviceBCOOMatrix& other) = default;
    DeviceBCOOMatrix& operator=(DeviceBCOOMatrix&& other)      = default;

    auto viewer()
    {
        return BCOOMatrixViewer<T, N>{m_block_rows,
                                      m_block_cols,
                                      0,
                                      (int)m_block_values.size(),
                                      (int)m_block_values.size(),
                                      m_block_row_indices.data(),
                                      m_block_col_indices.data(),
                                      m_block_values.data()};
    }

    auto cviewer() const
    {
        return BCOOMatrixViewer<T, N>{m_block_rows,
                                      m_block_cols,
                                      0,
                                      (int)m_block_values.size(),
                                      (int)m_block_values.size(),
                                      m_block_row_indices.data(),
                                      m_block_col_indices.data(),
                                      m_block_values.data()};
    }

    auto non_zero_blocks() const { return m_block_values.size(); }
};

template <typename T>
class DeviceBCOOMatrix<T, 1> : public DeviceTripletMatrix<T, 1>
{
    template <typename T, int N>
    friend class details::MatrixFormatConverter;

  public:
    using T = double;  // for flexibility

    DeviceBCOOMatrix() = default;
    ~DeviceBCOOMatrix() { destroy_all_descr(); }

    DeviceBCOOMatrix(const DeviceBCOOMatrix& other)
        : DeviceTripletMatrix{other}
        , m_legacy_descr{nullptr}
        , m_descr{nullptr}
    {
    }

    DeviceBCOOMatrix(DeviceBCOOMatrix&& other)
        : DeviceTripletMatrix{std::move(other)}
        , m_legacy_descr{other.m_legacy_descr}
        , m_descr{other.m_descr}
    {
        other.m_legacy_descr = nullptr;
        other.m_descr        = nullptr;
    }

    DeviceBCOOMatrix& operator=(const DeviceBCOOMatrix& other)
    {
        if(this == &other)
            return *this;
        DeviceTripletMatrix::operator=(other);
        destroy_all_descr();
        m_legacy_descr = nullptr;
        m_descr        = nullptr;
        return *this;
    }

    DeviceBCOOMatrix& operator=(DeviceBCOOMatrix&& other)
    {
        if(this == &other)
            return *this;
        DeviceTripletMatrix::operator=(std::move(other));
        destroy_all_descr();
        m_legacy_descr       = other.m_legacy_descr;
        m_descr              = other.m_descr;
        other.m_legacy_descr = nullptr;
        other.m_descr        = nullptr;
        return *this;
    }

    mutable cusparseMatDescr_t   m_legacy_descr = nullptr;
    mutable cusparseSpMatDescr_t m_descr        = nullptr;

    auto viewer()
    {
        return COOMatrixViewer{m_rows,
                               m_cols,
                               0,
                               (int)m_values.size(),
                               (int)m_values.size(),
                               m_row_indices.data(),
                               m_col_indices.data(),
                               m_values.data()};
    }

    auto cviewer() const
    {
        return CCOOMatrixViewer{m_rows,
                                m_cols,
                                0,
                                (int)m_values.size(),
                                (int)m_values.size(),
                                m_row_indices.data(),
                                m_col_indices.data(),
                                m_values.data()};
    }


    auto non_zeros() const { return m_values.size(); }

    auto legacy_descr() const
    {
        if(m_legacy_descr == nullptr)
        {
            checkCudaErrors(cusparseCreateMatDescr(&m_legacy_descr));
            checkCudaErrors(cusparseSetMatType(m_legacy_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            checkCudaErrors(cusparseSetMatIndexBase(m_legacy_descr, CUSPARSE_INDEX_BASE_ZERO));
        }
        return m_legacy_descr;
    }

    auto descr() const
    {
        if(m_descr == nullptr)
        {
            checkCudaErrors(cusparseCreateCoo(&m_descr,
                                              m_rows,
                                              m_cols,
                                              non_zeros(),
                                              (void*)m_row_indices.data(),
                                              (void*)m_col_indices.data(),
                                              (void*)m_values.data(),
                                              CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO,
                                              cuda_data_type<T>()));
        }
        return m_descr;
    }

  private:
    void destroy_all_descr()
    {
        if(m_legacy_descr != nullptr)
        {
            checkCudaErrors(cusparseDestroyMatDescr(m_legacy_descr));
            m_legacy_descr = nullptr;
        }
        if(m_descr != nullptr)
        {
            checkCudaErrors(cusparseDestroySpMat(m_descr));
            m_descr = nullptr;
        }
    }
};

template <typename T>
using DeviceCOOMatrix = DeviceBCOOMatrix<T, 1>;
}  // namespace muda

#include "details/device_bcoo_matrix.inl"
