#pragma once
#include <muda/buffer/device_buffer.h>
#include <cusparse.h>

namespace muda::details
{
template <typename T, int N>
class MatrixFormatConverter;
}

namespace muda
{
template <typename T>
class DeviceCSRMatrix
{
    template <typename T, int N>
    friend class details::MatrixFormatConverter;

  public:
    muda::DeviceBuffer<double>   m_values;
    muda::DeviceBuffer<int>      m_row_offsets;
    muda::DeviceBuffer<int>      m_col_indices;
    mutable cusparseSpMatDescr_t m_descr        = nullptr;
    mutable cusparseMatDescr_t   m_legacy_descr = nullptr;

    int m_row = 0;
    int m_col = 0;

  public:
    DeviceCSRMatrix() = default;
    ~DeviceCSRMatrix();

    DeviceCSRMatrix(const DeviceCSRMatrix&);
    DeviceCSRMatrix(DeviceCSRMatrix&&) noexcept;

    DeviceCSRMatrix& operator=(const DeviceCSRMatrix&);
    DeviceCSRMatrix& operator=(DeviceCSRMatrix&&) noexcept;

    void reshape(int row, int col);

    auto values() { return m_values.view(); }
    auto values() const { return m_values.view(); }

    auto row_offsets() { return m_row_offsets.view(); }
    auto row_offsets() const { return m_row_offsets.view(); }

    auto col_indices() { return m_col_indices.view(); }
    auto col_indices() const { return m_col_indices.view(); }

    auto rows() const { return m_row; }
    auto cols() const { return m_col; }
    auto non_zeros() const { return m_values.size(); }

    cusparseSpMatDescr_t descr() const;
    cusparseMatDescr_t   legacy_descr() const;

  private:
    void destroy_all_descr() const;
};
}  // namespace muda
#include "details/device_csr_matrix.inl"
