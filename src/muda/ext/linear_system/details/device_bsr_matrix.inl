#include <muda/check/check_cusparse.h>
#include <muda/ext/linear_system/type_mapper/data_type_mapper.h>
namespace muda
{
template <typename T, int N>
DeviceBSRMatrix<T, N>::~DeviceBSRMatrix()
{
    if(m_legacy_descr)
    {
        checkCudaErrors(cusparseDestroyMatDescr(m_legacy_descr));
    }
    if(m_descr)
    {
        checkCudaErrors(cusparseDestroySpMat(m_descr));
    }
}

template <typename T, int N>
DeviceBSRMatrix<T, N>::DeviceBSRMatrix(const DeviceBSRMatrix& other)
    : m_row(other.m_row)
    , m_col(other.m_col)
    , m_block_row_offsets(other.m_block_row_offsets)
    , m_block_col_indices(other.m_block_col_indices)
    , m_block_values(other.m_block_values)
{
}

template <typename T, int N>
DeviceBSRMatrix<T, N>::DeviceBSRMatrix(DeviceBSRMatrix&& other)
    : m_row(other.m_row)
    , m_col(other.m_col)
    , m_block_row_offsets(std::move(other.m_block_row_offsets))
    , m_block_col_indices(std::move(other.m_block_col_indices))
    , m_block_values(std::move(other.m_block_values))
    , m_legacy_descr(other.m_legacy_descr)
{
    other.m_row          = 0;
    other.m_col          = 0;
    other.m_legacy_descr = nullptr;
    other.m_descr        = nullptr;
}

template <typename T, int N>
DeviceBSRMatrix<T, N>& DeviceBSRMatrix<T, N>::operator=(const DeviceBSRMatrix& other)
{
    if(this != &other)
    {
        m_row               = other.m_row;
        m_col               = other.m_col;
        m_block_row_offsets = other.m_block_row_offsets;
        m_block_col_indices = other.m_block_col_indices;
        m_block_values      = other.m_block_values;
        m_legacy_descr      = nullptr;
        m_descr             = nullptr;
    }
    return *this;
}

template <typename T, int N>
DeviceBSRMatrix<T, N>& DeviceBSRMatrix<T, N>::operator=(DeviceBSRMatrix&& other)
{
    if(this != &other)
    {
        m_row               = other.m_row;
        m_col               = other.m_col;
        m_block_row_offsets = std::move(other.m_block_row_offsets);
        m_block_col_indices = std::move(other.m_block_col_indices);
        m_block_values      = std::move(other.m_block_values);
        m_legacy_descr      = other.m_legacy_descr;
        m_descr             = other.m_descr;

        other.m_row          = 0;
        other.m_col          = 0;
        other.m_legacy_descr = nullptr;
        other.m_descr        = nullptr;
    }
    return *this;
}

template <typename T, int N>
void DeviceBSRMatrix<T, N>::reshape(int row, int col)
{
    m_row = row;
    m_block_row_offsets.resize(row + 1);
    m_col   = col;
    m_descr = nullptr;
}
template <typename T, int N>
cusparseMatDescr_t DeviceBSRMatrix<T, N>::legacy_descr() const
{
    if(m_legacy_descr == nullptr)
    {
        checkCudaErrors(cusparseCreateMatDescr(&m_legacy_descr));
    }
    return m_legacy_descr;
}

template <typename T, int N>
cusparseSpMatDescr_t DeviceBSRMatrix<T, N>::descr() const
{
    if(m_descr == nullptr)
    {
        //checkCudaErrors(cusparseCreateMatDescr(&m_legacy_descr));
        checkCudaErrors(cusparseCreateBsr(
            &m_descr,
            m_row,
            m_col,
            m_block_values.size(),
            N,
            N,
            remove_const(m_block_row_offsets.data()),
            remove_const(m_block_col_indices.data()),
            remove_const(m_block_values.data()),
            cusparse_index_type<decltype(m_block_row_offsets)::value_type>(),
            cusparse_index_type<decltype(m_block_col_indices)::value_type>(),
            CUSPARSE_INDEX_BASE_ZERO,
            cuda_data_type<T>(),
            cusparseOrder_t::CUSPARSE_ORDER_COL));
    }
    return m_descr;
}
}  // namespace muda
