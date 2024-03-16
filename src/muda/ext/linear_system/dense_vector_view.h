#pragma once
#include <cusparse_v2.h>
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

    using Base = ViewBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

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
    auto_const_t<T>*             m_data;
    mutable cusparseDnVecDescr_t m_descr;
    int                          m_offset;
    int                          m_inc;
    int                          m_size;
    int                          m_origin_size;

  public:
    DenseVectorViewBase(auto_const_t<T>*     data,
                        cusparseDnVecDescr_t descr,
                        int                  offset,
                        int                  inc,
                        int                  size,
                        int                  origin_size)
        : m_data(data)
        , m_descr(descr)
        , m_offset(offset)
        , m_inc(inc)
        , m_size(size)
        , m_origin_size(origin_size)
    {
    }

    ConstView as_const() const
    {
        return ConstView{m_data, m_descr, m_offset, m_inc, m_size, m_origin_size};
    }
    operator ConstView() const { return as_const(); }

    // non-const accessor
    auto viewer()
    {
        return ThisViewer{m_data, m_offset, m_size, m_origin_size};
    }
    auto buffer_view()
    {
        return ThisBufferView{m_data, size_t(m_offset), size_t(m_inc * m_size)};
    }

    auto data() { return m_data + m_offset; }
    auto origin_data() { return m_data; }

    // const accessor
    auto offset() const { return m_offset; }
    auto size() const { return m_size; }
    auto data() const { return m_data + m_offset; }
    auto origin_data() const { return m_data; }

    CBufferView buffer_view() const
    {
        return remove_const(*this).buffer_view();
    }

    auto cviewer() const
    {
        MUDA_KERNEL_ASSERT(inc() == 1, "When using cviewer(), inc!=1 is not allowed");
        return CViewer{m_data, m_offset, m_size, m_origin_size};
    }

    auto inc() const { return m_inc; }

    auto descr() const
    {
        MUDA_KERNEL_ASSERT(inc() == 1, "When using descr(), inc!=1 is not allowed");
        return m_descr;
    }

    auto subview(int offset, int size)
    {
        MUDA_KERNEL_ASSERT(inc() == 1, "When using subview(), inc!=1 is not allowed");
        MUDA_KERNEL_ASSERT(offset + size <= m_size, "subview out of range");
        return ThisView{m_data, m_descr, m_offset + offset, m_inc, size, m_origin_size};
    }

    auto subview(int offset, int size) const
    {
        MUDA_KERNEL_ASSERT(inc() == 1, "When using subview(), inc!=1 is not allowed");
        MUDA_KERNEL_ASSERT(offset + size <= m_size, "subview out of range");
        return ConstView{m_data, m_descr, m_offset + offset, m_inc, size, m_origin_size};
    }
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
