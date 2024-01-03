#pragma once
#include <muda/ext/linear_system/dense_vector_viewer.h>
#include <muda/buffer/buffer_view.h>

namespace muda
{
template <typename T>
class DenseVectorViewBase
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "now only support real number");

  protected:
    BufferView<T> m_view;
    int           m_inc;

  public:
    DenseVectorViewBase(const BufferView<T>& view, int inc = 1)
        : m_view(view)
        , m_inc(inc)

    {
    }

    auto                  size() const { return m_view.size(); }
    auto                  data() const { return m_view.data(); }
    CBufferView<T>        buffer_view() const { return m_view; }
    CDenseVectorViewer<T> cviewer() const;
    auto                  inc() const { return m_inc; }
};

template <typename T>
class CDenseVectorView : public DenseVectorViewBase<T>
{
    using Base = DenseVectorViewBase<T>;

  public:
    CDenseVectorView(const CBufferView<T>& view, int inc = 1)
        : Base(BufferViewBase<T>{view}, inc)
    {
    }

    CDenseVectorView(const Base& base)
        : Base(base)
    {
    }
};

template <typename T>
class DenseVectorView : public DenseVectorViewBase<T>
{
    using Base = DenseVectorViewBase<T>;

  public:
    using Base::Base;
    DenseVectorView(const Base& base)
        : Base(base)
    {
    }

    DenseVectorView(const CDenseVectorView<T>&) = delete;

    DenseVectorViewer<T> viewer();

    using Base::data;
    auto data() { return m_view.data(); }
    auto buffer_view() const { return m_view; }
};
}  // namespace muda

#include "details/dense_vector_view.inl"
