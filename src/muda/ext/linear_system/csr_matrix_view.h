#pragma once
#include <muda/ext/linear_system/common.h>
#include <muda/view/view_base.h>
namespace muda
{
template <bool IsConst, typename Ty>
class CSRMatrixViewBase : public ViewBase<IsConst>
{
    using Base = ViewBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    static_assert(!std::is_const_v<Ty>, "Ty must be non-const");
    using ConstView    = CSRMatrixViewBase<true, Ty>;
    using NonConstView = CSRMatrixViewBase<false, Ty>;
    using ThisView     = CSRMatrixViewBase<IsConst, Ty>;

  protected:
    // data
    int m_row = 0;
    int m_col = 0;


    auto_const_t<int>* m_row_offsets = nullptr;
    auto_const_t<int>* m_col_indices = nullptr;
    auto_const_t<Ty>*  m_values      = nullptr;
    int                m_non_zero    = 0;

    mutable cusparseSpMatDescr_t m_descr        = nullptr;
    mutable cusparseMatDescr_t   m_legacy_descr = nullptr;


    bool m_trans = false;

  public:
    CSRMatrixViewBase() = default;
    CSRMatrixViewBase(int                  row,
                      int                  col,
                      auto_const_t<int>*   row_offsets,
                      auto_const_t<int>*   col_indices,
                      auto_const_t<Ty>*    values,
                      int                  non_zero,
                      cusparseSpMatDescr_t descr,
                      cusparseMatDescr_t   legacy_descr,
                      bool                 trans)
        : m_row(row)
        , m_col(col)
        , m_row_offsets(row_offsets)
        , m_col_indices(col_indices)
        , m_values(values)
        , m_non_zero(non_zero)
        , m_descr(descr)
        , m_legacy_descr(legacy_descr)
        , m_trans(trans)
    {
    }

    ConstView as_const() const
    {
        return ConstView{
            m_row, m_col, m_row_offsets, m_col_indices, m_values, m_non_zero, m_descr, m_legacy_descr, m_trans};
    }

    // implicit conversion to const
    operator ConstView() const { return as_const(); }

    auto_const_t<Ty>*  values() { return m_values; }
    auto_const_t<int>* row_offsets() { return m_row_offsets; }
    auto_const_t<int>* col_indices() { return m_col_indices; }

    auto values() const { return m_values; }
    auto row_offsets() const { return m_row_offsets; }
    auto col_indices() const { return m_col_indices; }
    auto rows() const { return m_row; }
    auto cols() const { return m_col; }
    auto non_zeros() const { return m_non_zero; }
    auto descr() const { return m_descr; }
    auto legacy_descr() const { return m_legacy_descr; }
    auto is_trans() const { return m_trans; }
    auto T() const
    {
        return ThisView{
            m_row, m_col, m_row_offsets, m_col_indices, m_values, m_non_zero, m_descr, m_legacy_descr, !m_trans};
    }
};

template <typename Ty>
using CSRMatrixView = CSRMatrixViewBase<false, Ty>;
template <typename Ty>
using CCSRMatrixView = CSRMatrixViewBase<true, Ty>;
}  // namespace muda

namespace muda
{
template <typename Ty>
struct read_only_viewer<CSRMatrixView<Ty>>
{
    using type = CCSRMatrixView<Ty>;
};

template <typename Ty>
struct read_write_viewer<CSRMatrixView<Ty>>
{
    using type = CSRMatrixView<Ty>;
};
}  // namespace muda


#include "details/csr_matrix_view.inl"
