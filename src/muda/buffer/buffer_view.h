/*****************************************************************/ /**
 * \file   buffer_view.h
 * \brief  A view interface for any array-like liner memory, which can be 
 * constructed from DeviceBuffer/DeviceVector or any thing that is a array-like
 * liner memory, e.g. raw cuda pointer.
 * 
 * \author MuGdxy
 * \date   January 2024
 *********************************************************************/
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cinttypes>
#include <muda/type_traits/type_modifier.h>
#include <muda/viewer/dense/dense_1d.h>
#include <muda/buffer/buffer_fwd.h>
#include <muda/view/view_base.h>

namespace muda
{
template <bool IsConst, typename T>
class BufferViewBase : public ViewBase<IsConst>
{
    using Base = ViewBase<IsConst>;
  public:
    static_assert(!std::is_const_v<T>, "Ty must be non-const");
    using ConstView    = BufferViewBase<true, T>;
    using NonConstView = BufferViewBase<false, T>;
    using ThisView     = BufferViewBase<IsConst, T>;

    using CViewer    = CDense1D<T>;
    using Viewer     = Dense1D<T>;
    using ThisViewer = std::conditional_t<IsConst, CViewer, Viewer>;

    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  protected:
    auto_const_t<T>* m_data   = nullptr;
    size_t           m_offset = ~0;
    size_t           m_size   = ~0;

  public:
    MUDA_GENERIC BufferViewBase() MUDA_NOEXCEPT = default;
    MUDA_GENERIC BufferViewBase(auto_const_t<T>* data, size_t offset, size_t size) MUDA_NOEXCEPT
        : m_data(data),
          m_offset(offset),
          m_size(size)
    {
    }
    MUDA_GENERIC BufferViewBase(auto_const_t<T>* data, size_t size) MUDA_NOEXCEPT
        : BufferViewBase(data, 0, size)
    {
    }

    MUDA_GENERIC ConstView as_const() const MUDA_NOEXCEPT
    {
        return ConstView{m_data, m_offset, m_size};
    }

    MUDA_GENERIC operator ConstView() const MUDA_NOEXCEPT { return as_const(); }

    // non-const accessor
    MUDA_GENERIC auto_const_t<T>* data() MUDA_NOEXCEPT
    {
        return m_data + m_offset;
    }

    MUDA_GENERIC auto_const_t<T>* data(size_t i) MUDA_NOEXCEPT
    {
        i += m_offset;
        return m_data + i;
    }

    MUDA_GENERIC auto_const_t<T>* origin_data() MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC ThisView subview(size_t offset, size_t size = ~0) MUDA_NOEXCEPT;
    MUDA_GENERIC ThisViewer viewer() MUDA_NOEXCEPT;

    // const accessor

    MUDA_GENERIC size_t   size() const MUDA_NOEXCEPT { return m_size; }
    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT
    {
        return remove_const(*this).data();
    }
    MUDA_GENERIC const T* data(size_t i) const MUDA_NOEXCEPT
    {
        return remove_const(*this).data(i);
    }
    MUDA_GENERIC const T* origin_data() const MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC size_t   offset() const MUDA_NOEXCEPT { return m_offset; }

    MUDA_GENERIC ConstView subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT;
    MUDA_GENERIC CViewer cviewer() const MUDA_NOEXCEPT;


    MUDA_GENERIC auto_const_t<T>& operator[](size_t i) MUDA_NOEXCEPT
    {
        return *data(i);
    }

    MUDA_GENERIC const T& operator[](size_t i) const MUDA_NOEXCEPT
    {
        return *data(i);
    }

    /**********************************************************************************
    * BufferView As Iterator
    ***********************************************************************************/

    // Random Access Iterator Interface
    using value_type        = T;
    using reference         = T&;
    using pointer           = T*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = size_t;

    MUDA_GENERIC ThisView operator+(int i)
    {
        return ThisView{m_data, m_offset + i, m_size - i};
    }
    MUDA_GENERIC ConstView operator+(int i) const
    {
        return remove_const(*this).operator+(i).as_const();
    }
    MUDA_GENERIC reference operator*() { return *data(0); }
    MUDA_GENERIC auto_const_t<T>& operator[](int i) { return *data(i); }
    MUDA_GENERIC const T&         operator[](int i) const { return *data(i); }
};

template <typename T>
class CBufferView : public BufferViewBase<true, T>
{
    using Base = BufferViewBase<true, T>;

  public:
    using Base::Base;

    MUDA_GENERIC CBufferView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CBufferView<T> subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT
    {
        return CBufferView{Base::subview(offset, size)};
    }

    MUDA_GENERIC CBufferView<T> subview(size_t offset, size_t size = ~0) MUDA_NOEXCEPT
    {
        return CBufferView{Base::subview(offset, size)};
    }

    MUDA_HOST void copy_to(T* host) const;

    MUDA_GENERIC auto as_const() const MUDA_NOEXCEPT { return *this; }
};

template <typename T>
class BufferView : public BufferViewBase<false, T>
{
    using Base = BufferViewBase<false, T>;

  public:
    using Base::BufferViewBase;
    using Base::data;
    using Base::origin_data;

    MUDA_GENERIC BufferView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC BufferView(const CBufferView<T>&) = delete;

    MUDA_GENERIC BufferView(Dense1D<T> viewer) MUDA_NOEXCEPT
        : Base(viewer.data(), 0, (size_t)viewer.total_size())
    {
    }

    MUDA_GENERIC CBufferView<T> as_const() const MUDA_NOEXCEPT
    {
        return CBufferView<T>{Base::as_const()};
    }

    MUDA_GENERIC operator CBufferView<T>() const MUDA_NOEXCEPT
    {
        return as_const();
    }

    MUDA_GENERIC BufferView<T> subview(size_t offset, size_t size = ~0) MUDA_NOEXCEPT
    {
        return BufferView{Base::subview(offset, size)};
    }

    MUDA_GENERIC CBufferView<T> subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT
    {
        return CBufferView{Base::subview(offset, size)};
    }

    MUDA_HOST void fill(const T& v);
    MUDA_HOST void copy_from(CBufferView<T> other);
    MUDA_HOST void copy_from(const T* host);
    MUDA_HOST void copy_to(T* host) const
    {
        CBufferView<T>{*this}.copy_to(host);
    }
};

template <typename T>
struct read_only_viewer<BufferView<T>>
{
    using type = CBufferView<T>;
};

template <typename T>
struct read_write_viewer<CBufferView<T>>
{
    using type = BufferView<T>;
};
}  // namespace muda

#include "details/buffer_view.inl"