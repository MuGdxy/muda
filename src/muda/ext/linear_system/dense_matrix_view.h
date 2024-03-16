#pragma once
#include <muda/ext/linear_system/dense_matrix_viewer.h>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/view/view_base.h>
namespace muda
{
template <bool IsConst, typename Ty>
class DenseMatrixViewBase : public ViewBase<IsConst>
{
    using Base = ViewBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    static_assert(std::is_same_v<Ty, float> || std::is_same_v<Ty, double>,
                  "now only support real number");
    static_assert(!std::is_const_v<Ty>, "Ty must be non-const");
    using ConstView    = DenseMatrixViewBase<true, Ty>;
    using NonConstView = DenseMatrixViewBase<false, Ty>;
    using ThisView     = DenseMatrixViewBase<IsConst, Ty>;

    using CBuffer2DView = CBuffer2DView<Ty>;
    using Buffer2DView  = Buffer2DView<Ty>;
    using ThisBuffer2DView = std::conditional_t<IsConst, CBuffer2DView, Buffer2DView>;

    using CViewer    = CDenseMatrixViewer<Ty>;
    using Viewer     = DenseMatrixViewer<Ty>;
    using ThisViewer = std::conditional_t<IsConst, CViewer, Viewer>;

  protected:
    ThisBuffer2DView m_view;
    size_t           m_row   = 0;
    size_t           m_col   = 0;
    bool             m_trans = false;
    bool             m_sym   = false;

  public:
    MUDA_GENERIC DenseMatrixViewBase(ThisBuffer2DView view,
                                     size_t           row,
                                     size_t           col,
                                     bool             trans = false,
                                     bool             sym = false) MUDA_NOEXCEPT
        : m_view(view),
          m_row(row),
          m_col(col),
          m_trans(trans),
          m_sym(sym)
    {
    }

    // implicit conversion
    MUDA_GENERIC auto as_const() const MUDA_NOEXCEPT
    {
        return ConstView{m_view, m_row, m_col, m_trans, m_sym};
    }
    MUDA_GENERIC operator ConstView() const MUDA_NOEXCEPT { return as_const(); }

    // non-const accessor
    MUDA_GENERIC auto     data() MUDA_NOEXCEPT { return m_view.origin_data(); }
    MUDA_GENERIC ThisView T() MUDA_NOEXCEPT;
    MUDA_GENERIC ThisViewer viewer() MUDA_NOEXCEPT;
    MUDA_GENERIC auto       buffer_view() MUDA_NOEXCEPT { return m_view; }

    // const accessor
    MUDA_GENERIC bool   is_trans() const MUDA_NOEXCEPT { return m_trans; }
    MUDA_GENERIC bool   is_sym() const MUDA_NOEXCEPT { return m_sym; }
    MUDA_GENERIC size_t row() const MUDA_NOEXCEPT { return m_row; }
    MUDA_GENERIC size_t col() const MUDA_NOEXCEPT { return m_col; }
    MUDA_GENERIC size_t lda() const MUDA_NOEXCEPT
    {
        return m_view.pitch_bytes() / sizeof(Ty);
    }
    MUDA_GENERIC ConstView T() const MUDA_NOEXCEPT
    {
        return remove_const(*this).T();
    }
    MUDA_GENERIC auto data() const MUDA_NOEXCEPT
    {
        return m_view.origin_data();
    }
    MUDA_GENERIC CViewer cviewer() const MUDA_NOEXCEPT
    {
        return remove_const(*this).viewer();
    }
    MUDA_GENERIC CBuffer2DView buffer_view() const MUDA_NOEXCEPT
    {
        return m_view;
    }
};

template <typename Ty>
class CDenseMatrixView : public DenseMatrixViewBase<true, Ty>
{
    using Base = DenseMatrixViewBase<true, Ty>;

  public:
    using Base::Base;
    CDenseMatrixView(const Base& base)
        : Base(base)
    {
    }

    CDenseMatrixView<Ty> T() const { return Base::T(); }
};

template <typename Ty>
class DenseMatrixView : public DenseMatrixViewBase<false, Ty>
{
    using Base = DenseMatrixViewBase<false, Ty>;

  public:
    using Base::Base;

    DenseMatrixView(const Base& base)
        : Base(base)
    {
    }

    DenseMatrixView(const CDenseMatrixView<Ty>&) = delete;

    auto T() const { return Base::T(); }
};
}  // namespace muda

#include "details/dense_matrix_view.inl"
