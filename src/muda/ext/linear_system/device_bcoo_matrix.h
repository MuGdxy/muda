#pragma once
#include <muda/buffer/device_buffer.h>
#include <muda/ext/linear_system/bcoo_matrix_view.h>
#include <muda/ext/linear_system/device_triplet_matrix.h>
#include <cusparse.h>
#include <muda/ext/linear_system/type_mapper/data_type_mapper.h>

namespace muda::details
{
    template <typename T, int N>
    class MatrixFormatConverter;
}

namespace muda
{
template <typename T, int N>
class DeviceBCOOMatrix : public DeviceTripletMatrix<T, N>
{
    friend class details::MatrixFormatConverter<T, N>;

  public:
    using BlockMatrix = Eigen::Matrix<T, N, N>;

    DeviceBCOOMatrix()                                   = default;
    ~DeviceBCOOMatrix()                                  = default;
    DeviceBCOOMatrix(const DeviceBCOOMatrix&)            = default;
    DeviceBCOOMatrix(DeviceBCOOMatrix&&)                 = default;
    DeviceBCOOMatrix& operator=(const DeviceBCOOMatrix&) = default;
    DeviceBCOOMatrix& operator=(DeviceBCOOMatrix&&)      = default;
    auto non_zero_blocks() const { return this->m_block_values.size(); }
};

template <typename Ty>
class DeviceBCOOMatrix<Ty, 1> : public DeviceTripletMatrix<Ty, 1>
{
    template <typename U, int M>
    friend class details::MatrixFormatConverter;

  protected:
    mutable cusparseMatDescr_t   m_legacy_descr = nullptr;
    mutable cusparseSpMatDescr_t m_descr        = nullptr;

  public:
    DeviceBCOOMatrix() = default;
    ~DeviceBCOOMatrix() { destroy_all_descr(); }

    DeviceBCOOMatrix(const DeviceBCOOMatrix& other)
        : DeviceTripletMatrix<Ty, 1>{other}
        , m_legacy_descr{nullptr}
        , m_descr{nullptr}
    {
    }

    DeviceBCOOMatrix(DeviceBCOOMatrix&& other)
        : DeviceTripletMatrix<Ty, 1>{std::move(other)}
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
        DeviceTripletMatrix<Ty, 1>::operator=(other);
        destroy_all_descr();
        m_legacy_descr = nullptr;
        m_descr        = nullptr;
        return *this;
    }

    DeviceBCOOMatrix& operator=(DeviceBCOOMatrix&& other)
    {
        if(this == &other)
            return *this;
        DeviceTripletMatrix<Ty, 1>::operator=(std::move(other));
        destroy_all_descr();
        m_legacy_descr       = other.m_legacy_descr;
        m_descr              = other.m_descr;
        other.m_legacy_descr = nullptr;
        other.m_descr        = nullptr;
        return *this;
    }


    auto view()
    {
        return COOMatrixView<Ty>{this->m_rows,
                                 this->m_cols,
                                 (int)this->m_values.size(),
                                 this->m_row_indices.data(),
                                 this->m_col_indices.data(),
                                 this->m_values.data(),
                                 descr(),
                                 legacy_descr(),
                                 false};
    }

    auto view() const
    {
        return CCOOMatrixView<Ty>{this->m_rows,
                                  this->m_cols,
                                  (int)this->m_values.size(),
                                  this->m_row_indices.data(),
                                  this->m_col_indices.data(),
                                  this->m_values.data(),
                                  descr(),
                                  legacy_descr(),
                                  false};
    }

    auto cview() const { return view(); }

    auto viewer() { return view().viewer(); }

    auto cviewer() const { return view().cviewer(); }

    auto non_zeros() const { return this->m_values.size(); }

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
                                              this->m_rows,
                                              this->m_cols,
                                              non_zeros(),
                                              (void*)this->m_row_indices.data(),
                                              (void*)this->m_col_indices.data(),
                                              (void*)this->m_values.data(),
                                              CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO,
                                              cuda_data_type<Ty>()));
        }
        return m_descr;
    }

    //auto T() const { return view().T(); }
    //auto T() { return view().T(); }

    operator COOMatrixView<Ty>() { return view(); }
    operator CCOOMatrixView<Ty>() const { return view(); }

    void clear()
    {
        DeviceTripletMatrix<Ty, 1>::clear();
        destroy_all_descr();
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
