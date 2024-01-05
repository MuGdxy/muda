#pragma once

namespace muda
{
template <typename Ty, int N>
class BSRMatrixViewBase
{
    static_assert(N >= 2, "Block size must be >= 2");

  public:
    using BlockMatrix = Eigen::Matrix<Ty, N, N>;

  protected:
    int m_row = 0;
    int m_col = 0;

    int*         m_block_row_offsets = nullptr;
    int*         m_block_col_indices = nullptr;
    BlockMatrix* m_block_values      = nullptr;
    size_t       m_non_zeros         = 0;

    cusparseMatDescr_t   m_legacy_descr = nullptr;
    cusparseSpMatDescr_t m_descr        = nullptr;

    bool m_trans = false;


  public:
    BSRMatrixViewBase() = default;
    BSRMatrixViewBase(int                  row,
                      int                  col,
                      int*                 block_row_offsets,
                      int*                 block_col_indices,
                      BlockMatrix*         block_values,
                      size_t               non_zero_blocks,
                      cusparseMatDescr_t   legacy_descr,
                      cusparseSpMatDescr_t descr,
                      bool                 trans) MUDA_NOEXCEPT
        : m_row(row),
          m_col(col),
          m_block_row_offsets(block_row_offsets),
          m_block_col_indices(block_col_indices),
          m_non_zeros(non_zero_blocks),
          m_block_values(block_values),
          m_legacy_descr(legacy_descr),
          m_descr(descr),
          m_trans(trans)
    {
    }

    auto T() const
    {
        return BSRMatrixViewBase{m_row,
                                 m_col,
                                 m_block_row_offsets,
                                 m_block_col_indices,
                                 m_block_values,
                                 m_non_zeros,
                                 m_legacy_descr,
                                 m_descr,
                                 !m_trans};
    }

    auto block_values() { return m_block_values; }
    auto block_values() const { return m_block_values; }

    auto block_row_offsets() { return m_block_row_offsets; }
    auto block_row_offsets() const { return m_block_row_offsets; }

    auto block_col_indices() { return m_block_col_indices; }
    auto block_col_indices() const { return m_block_col_indices; }

    auto block_rows() const { return m_row; }
    auto block_cols() const { return m_col; }
    auto non_zero_blocks() const { return m_non_zeros; }

    auto legacy_descr() const { return m_legacy_descr; }
    auto descr() const { return m_descr; }
};

template <typename Ty, int N>
class CBSRMatrixView : public BSRMatrixViewBase<Ty, N>
{
    using Base = BSRMatrixViewBase<Ty, N>;

  public:
    using Base::Base;
    CBSRMatrixView(int                  row,
                   int                  col,
                   const int*           block_row_offsets,
                   const int*           block_col_indices,
                   const BlockMatrix*   block_values,
                   size_t               non_zero_blocks,
                   cusparseMatDescr_t   legacy_descr,
                   cusparseSpMatDescr_t descr,
                   bool                 trans) MUDA_NOEXCEPT  //
        : Base(row,
               col,
               remove_const(block_row_offsets),
               remove_const(block_col_indices),
               remove_const(block_values),
               non_zero_blocks,
               legacy_descr,
               descr,
               trans)
    {
    }

    CBSRMatrixView(const Base& base) MUDA_NOEXCEPT : Base(base) {}

    auto T() const { return CBSRMatrixView{Base::T()}; }
};

template <typename Ty, int N>
class BSRMatrixView : public BSRMatrixViewBase<Ty, N>
{
    using Base = BSRMatrixViewBase<Ty, N>;

  public:
    using Base::Base;

    BSRMatrixView(const Base& base) MUDA_NOEXCEPT : Base(base) {}

    auto T() const { return BSRMatrixView{Base::T()}; }
};
}  // namespace muda
#include "details/bsr_matrix_view.inl"
