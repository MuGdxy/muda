#pragma once
#include <cusparse_v2.h>
#include <muda/view/view_base.h>
namespace muda
{
template <bool IsConst, typename Ty, int N>
class BSRMatrixViewBase : public ViewBase<IsConst>
{
    using Base = ViewBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    static_assert(!std::is_const_v<Ty>, "Ty must be non-const");
    using ConstView    = BSRMatrixViewBase<true, Ty, N>;
    using NonConstView = BSRMatrixViewBase<false, Ty, N>;
    using ThisView     = BSRMatrixViewBase<IsConst, Ty, N>;

    using BlockMatrix = Eigen::Matrix<Ty, N, N>;

  protected:
    // data
    int m_row = 0;
    int m_col = 0;

    auto_const_t<int>*         m_block_row_offsets = nullptr;
    auto_const_t<int>*         m_block_col_indices = nullptr;
    auto_const_t<BlockMatrix>* m_block_values      = nullptr;
    int                        m_non_zeros         = 0;

    mutable cusparseMatDescr_t   m_legacy_descr = nullptr;
    mutable cusparseSpMatDescr_t m_descr        = nullptr;

    bool m_trans = false;

  public:
    BSRMatrixViewBase() = default;
    BSRMatrixViewBase(int                        row,
                      int                        col,
                      auto_const_t<int>*         block_row_offsets,
                      auto_const_t<int>*         block_col_indices,
                      auto_const_t<BlockMatrix>* block_values,
                      int                        non_zeros,
                      cusparseSpMatDescr_t       descr,
                      cusparseMatDescr_t         legacy_descr,
                      bool                       trans)
        : m_row(row)
        , m_col(col)
        , m_block_row_offsets(block_row_offsets)
        , m_block_col_indices(block_col_indices)
        , m_block_values(block_values)
        , m_non_zeros(non_zeros)
        , m_descr(descr)
        , m_legacy_descr(legacy_descr)
        , m_trans(trans)

    {
    }

    // explicit conversion to non-const
    ConstView as_const() const
    {
        return ConstView{m_row,
                         m_col,
                         m_block_row_offsets,
                         m_block_col_indices,
                         m_block_values,
                         m_non_zeros,
                         m_descr,
                         m_legacy_descr,
                         m_trans};
    }

    // implicit conversion to const
    operator ConstView() const { return as_const(); }

    // non-const access
    auto_const_t<BlockMatrix>* block_values() { return m_block_values; }
    auto_const_t<int>* block_row_offsets() { return m_block_row_offsets; }
    auto_const_t<int>* block_col_indices() { return m_block_col_indices; }

    // const access
    auto block_values() const { return m_block_values; }
    auto block_row_offsets() const { return m_block_row_offsets; }
    auto block_col_indices() const { return m_block_col_indices; }

    auto block_rows() const { return m_row; }
    auto block_cols() const { return m_col; }
    auto non_zero_blocks() const { return m_non_zeros; }

    auto legacy_descr() const { return m_legacy_descr; }
    auto descr() const { return m_descr; }
    auto is_trans() const { return m_trans; }

    auto T() const
    {
        return ThisView{m_row,
                        m_col,
                        m_block_row_offsets,
                        m_block_col_indices,
                        m_block_values,
                        m_non_zeros,
                        m_descr,
                        m_legacy_descr,
                        !m_trans};
    }
};

template <typename Ty, int N>
using BSRMatrixView = BSRMatrixViewBase<false, Ty, N>;
template <typename Ty, int N>
using CBSRMatrixView = BSRMatrixViewBase<true, Ty, N>;
}  // namespace muda

namespace muda
{
template <typename Ty, int N>
struct read_only_viewer<BSRMatrixView<Ty, N>>
{
    using type = CBSRMatrixView<Ty, N>;
};

template <typename Ty, int N>
struct read_write_viewer<CBSRMatrixView<Ty, N>>
{
    using type = BSRMatrixView<Ty, N>;
};
}  // namespace muda


#include "details/bsr_matrix_view.inl"
