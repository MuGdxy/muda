#pragma once
#include <muda/buffer/device_buffer.h>
#include <muda/ext/linear_system/bsr_matrix_view.h>
#include <cusparse.h>

namespace muda::details
{
template <typename T, int N>
class MatrixFormatConverter;
}

namespace muda
{
template <typename Ty, int N>
class DeviceBSRMatrix
{
    friend class details::MatrixFormatConverter<Ty, N>;
    static_assert(N >= 2, "Block size must be >= 2");

  public:
    using BlockMatrix = Eigen::Matrix<Ty, N, N>;

  protected:
    muda::DeviceBuffer<BlockMatrix> m_block_values;
    muda::DeviceBuffer<int>         m_block_row_offsets;
    muda::DeviceBuffer<int>         m_block_col_indices;
    mutable cusparseSpMatDescr_t    m_descr        = nullptr;
    mutable cusparseMatDescr_t      m_legacy_descr = nullptr;

    int m_row = 0;
    int m_col = 0;

  public:
    DeviceBSRMatrix() = default;
    ~DeviceBSRMatrix();

    DeviceBSRMatrix(const DeviceBSRMatrix&);
    DeviceBSRMatrix(DeviceBSRMatrix&&);

    DeviceBSRMatrix& operator=(const DeviceBSRMatrix&);
    DeviceBSRMatrix& operator=(DeviceBSRMatrix&&);

    void reshape(int row, int col);
    void reserve(int non_zero_blocks);
    void resize(int non_zero_blocks);

    static constexpr int block_size() { return N; }

    auto block_values() { return m_block_values.view(); }
    auto block_values() const { return m_block_values.view(); }

    auto block_row_offsets() { return m_block_row_offsets.view(); }
    auto block_row_offsets() const { return m_block_row_offsets.view(); }

    auto block_col_indices() { return m_block_col_indices.view(); }
    auto block_col_indices() const { return m_block_col_indices.view(); }

    auto block_rows() const { return m_row; }
    auto block_cols() const { return m_col; }
    auto non_zero_blocks() const { return m_block_values.size(); }

    cusparseSpMatDescr_t descr() const;
    cusparseMatDescr_t   legacy_descr() const;

    auto view()
    {
        return BSRMatrixView<Ty, N>{m_row,
                                    m_col,
                                    m_block_row_offsets.data(),
                                    m_block_col_indices.data(),
                                    m_block_values.data(),
                                    (int)m_block_values.size(),
                                    descr(),
                                    legacy_descr(),
                                    false};
    }

    operator BSRMatrixView<Ty, N>() { return view(); }

    auto view() const
    {
        return CBSRMatrixView<Ty, N>{m_row,
                                     m_col,
                                     m_block_row_offsets.data(),
                                     m_block_col_indices.data(),
                                     m_block_values.data(),
                                     (int)m_block_values.size(),
                                     descr(),
                                     legacy_descr(),
                                     false};
    }

    operator CBSRMatrixView<Ty, N>() const { return view(); }

    auto cview() const { return view(); }

    auto T() const { return view().T(); }
    auto T() { return view().T(); }

    void clear();

  private:
    void destroy_all_descr() const;
};
}  // namespace muda

#include "details/device_bsr_matrix.inl"
