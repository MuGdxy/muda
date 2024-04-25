#pragma once
#include <muda/buffer/device_buffer.h>
#include <cusparse.h>
#include <muda/ext/linear_system/csr_matrix_view.h>

namespace muda::details
{
template <typename T, int N>
class MatrixFormatConverter;
}

namespace muda
{
template <typename Ty>
class DeviceCSRMatrix
{
    template <typename T, int N>
    friend class details::MatrixFormatConverter;

  public:
    int m_row = 0;
    int m_col = 0;

    muda::DeviceBuffer<int> m_row_offsets;
    muda::DeviceBuffer<int> m_col_indices;
    muda::DeviceBuffer<Ty>  m_values;

    mutable cusparseSpMatDescr_t m_descr        = nullptr;
    mutable cusparseMatDescr_t   m_legacy_descr = nullptr;

  public:
    DeviceCSRMatrix() = default;
    ~DeviceCSRMatrix();

    DeviceCSRMatrix(const DeviceCSRMatrix&);
    DeviceCSRMatrix(DeviceCSRMatrix&&) noexcept;

    DeviceCSRMatrix& operator=(const DeviceCSRMatrix&);
    DeviceCSRMatrix& operator=(DeviceCSRMatrix&&) noexcept;

    void reshape(int row, int col);
    void reserve(int non_zeros);

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

    auto view()
    {
        return CSRMatrixView<Ty>{m_row,
                                 m_col,
                                 m_row_offsets.data(),
                                 m_col_indices.data(),
                                 m_values.data(),
                                 (int)non_zeros(),
                                 descr(),
                                 legacy_descr(),
                                 false};
    }

    auto view() const
    {
        return CCSRMatrixView<Ty>{m_row,
                                  m_col,
                                  m_row_offsets.data(),
                                  m_col_indices.data(),
                                  m_values.data(),
                                  (int)non_zeros(),
                                  descr(),
                                  legacy_descr(),
                                  false};
    }

    auto cview() const { return view(); }

    auto T() const { return view().T(); }
    auto T() { return view().T(); }
    operator CSRMatrixView<Ty>() { return view(); }
    operator CCSRMatrixView<Ty>() const { return view(); }

    void clear();

  private:
    void destroy_all_descr() const;
};
}  // namespace muda
#include "details/device_csr_matrix.inl"
