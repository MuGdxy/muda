#include <muda/check/check_cusparse.h>
#include <muda/ext/linear_system/type_mapper/data_type_mapper.h>

namespace muda
{
template <typename Ty>
DeviceCSRMatrix<Ty>::~DeviceCSRMatrix()
{
    destroy_all_descr();
}

template <typename Ty>
DeviceCSRMatrix<Ty>::DeviceCSRMatrix(const DeviceCSRMatrix& other)
    : m_row(other.m_row)
    , m_col(other.m_col)
    , m_row_offsets(other.m_row_offsets)
    , m_col_indices(other.m_col_indices)
    , m_values(other.m_values)
{
}


template <typename Ty>
DeviceCSRMatrix<Ty>::DeviceCSRMatrix(DeviceCSRMatrix&& other) noexcept
    : m_row(other.m_row)
    , m_col(other.m_col)
    , m_row_offsets(std::move(other.m_row_offsets))
    , m_col_indices(std::move(other.m_col_indices))
    , m_values(std::move(other.m_values))
    , m_descr(other.m_descr)
{
    other.m_row          = 0;
    other.m_col          = 0;
    other.m_descr        = nullptr;
    other.m_legacy_descr = nullptr;
}


template <typename Ty>
DeviceCSRMatrix<Ty>& DeviceCSRMatrix<Ty>::operator=(const DeviceCSRMatrix& other)
{
    if(this != &other)
    {
        m_row         = other.m_row;
        m_col         = other.m_col;
        m_row_offsets = other.m_row_offsets;
        m_col_indices = other.m_col_indices;
        m_values      = other.m_values;
        destroy_all_descr();

        m_descr        = nullptr;
        m_legacy_descr = nullptr;
    }
    return *this;
}


template <typename Ty>
DeviceCSRMatrix<Ty>& DeviceCSRMatrix<Ty>::operator=(DeviceCSRMatrix&& other) noexcept
{
    if(this != &other)
    {
        m_row         = other.m_row;
        m_col         = other.m_col;
        m_row_offsets = std::move(other.m_row_offsets);
        m_col_indices = std::move(other.m_col_indices);
        m_values      = std::move(other.m_values);
        destroy_all_descr();

        m_descr        = other.m_descr;
        m_legacy_descr = other.m_legacy_descr;

        other.m_row          = 0;
        other.m_col          = 0;
        other.m_descr        = nullptr;
        other.m_legacy_descr = nullptr;
    }
    return *this;
}

template <typename Ty>
void DeviceCSRMatrix<Ty>::reshape(int row, int col)
{
    m_row = row;
    m_row_offsets.resize(row + 1);
    m_col = col;
    destroy_all_descr();
}

template <typename Ty>
void DeviceCSRMatrix<Ty>::reserve(int non_zeros)
{
    m_col_indices.reserve(non_zeros);
    m_values.reserve(non_zeros);
}

template <typename Ty>
cusparseSpMatDescr_t DeviceCSRMatrix<Ty>::descr() const
{
    if(m_descr == nullptr)
    {
        checkCudaErrors(cusparseCreateCsr(
            &m_descr,
            this->m_row,
            this->m_col,
            this->m_values.size(),
            remove_const(this->m_row_offsets.data()),
            remove_const(this->m_col_indices.data()),
            remove_const(this->m_values.data()),
            cusparse_index_type<typename decltype(this->m_row_offsets)::value_type>(),
            cusparse_index_type<typename decltype(this->m_col_indices)::value_type>(),
            CUSPARSE_INDEX_BASE_ZERO,
            cuda_data_type<Ty>()));
    }
    return m_descr;
}
template <typename Ty>
cusparseMatDescr_t DeviceCSRMatrix<Ty>::legacy_descr() const
{
    if(m_legacy_descr == nullptr)
    {
        checkCudaErrors(cusparseCreateMatDescr(&m_legacy_descr));
        checkCudaErrors(cusparseSetMatType(m_legacy_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        checkCudaErrors(cusparseSetMatIndexBase(m_legacy_descr, CUSPARSE_INDEX_BASE_ZERO));
        checkCudaErrors(cusparseSetMatDiagType(m_legacy_descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
    }
    return m_legacy_descr;
}
template <typename Ty>
void DeviceCSRMatrix<Ty>::clear()
{
    m_row = 0;
    m_col = 0;
    m_row_offsets.clear();
    m_col_indices.clear();
    m_values.clear();
    destroy_all_descr();
}

template <typename Ty>
void DeviceCSRMatrix<Ty>::destroy_all_descr() const
{
    if(m_descr)
    {
        checkCudaErrors(cusparseDestroySpMat(m_descr));
        m_descr = nullptr;
    }
    if(m_legacy_descr)
    {
        checkCudaErrors(cusparseDestroyMatDescr(m_legacy_descr));
        m_legacy_descr = nullptr;
    }
}
}  // namespace muda
