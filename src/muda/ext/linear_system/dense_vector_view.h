#pragma once
#include <muda/ext/linear_system/dense_vector_viewer.h>
#include <muda/buffer/buffer_view.h>
#include <muda/view/view_base.h>
namespace muda
{
template <bool IsConst, typename T>
class DenseVectorViewBase : public ViewBase<IsConst>
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "now only support real number");

  public:
    using NonConstView = DenseVectorViewBase<false, T>;
    using ConstView    = DenseVectorViewBase<true, T>;
    using ThisView     = DenseVectorViewBase<IsConst, T>;

    using CBufferView    = CBufferView<T>;
    using BufferView     = BufferView<T>;
    using ThisBufferView = std::conditional_t<IsConst, CBufferView, BufferView>;

    using CViewer    = CDenseVectorViewer<T>;
    using Viewer     = DenseVectorViewer<T>;
    using ThisViewer = std::conditional_t<IsConst, CViewer, Viewer>;

  protected:
    ThisBufferView m_view;
    int            m_inc;

  public:
    DenseVectorViewBase(ThisBufferView view, int inc = 1)
        : m_view(view)
        , m_inc(inc)

    {
    }

    ConstView as_const() const { return ConstView{m_view, m_inc}; }
    operator ConstView() const { return as_const(); }

    // non-const accessor
    auto viewer() { return ThisViewer{m_view, 0, m_view.size()}; }
    auto buffer_view() { return m_view; }

    // const accessor
    auto        size() const { return m_view.size(); }
    auto        data() const { return m_view.data(); }
    CBufferView buffer_view() const { return m_view; }
    auto        cviewer() const { return CViewer{m_view, 0, m_view.size()}; }
    auto        inc() const { return m_inc; }
};

template <typename Ty>
using DenseVectorView = DenseVectorViewBase<false, Ty>;
template <typename Ty>
using CDenseVectorView = DenseVectorViewBase<true, Ty>;
}  // namespace muda

namespace muda
{
template <typename Ty>
struct read_only_viewer<DenseVectorView<Ty>>
{
    using type = CDenseVectorView<Ty>;
};

template <typename Ty>
struct read_write_viewer<DenseVectorView<Ty>>
{
    using type = DenseVectorView<Ty>;
};
}  // namespace muda


#include "details/dense_vector_view.inl"
