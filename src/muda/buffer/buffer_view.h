#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cinttypes>
#include <muda/type_traits/type_modifier.h>
#include <muda/viewer/dense/dense_1d.h>
#include <muda/buffer/buffer_fwd.h>

namespace muda
{
template <typename T>
class BufferViewBase
{
    template <typename T, typename FConstruct>
    friend void resize(int              grid_dim,
                       int              block_dim,
                       cudaStream_t     stream,
                       DeviceBuffer<T>& buffer,
                       size_t           new_size,
                       FConstruct&&     fct);

  protected:
    T*     m_data   = nullptr;
    size_t m_offset = ~0;
    size_t m_size   = ~0;

  public:
    MUDA_GENERIC BufferViewBase() MUDA_NOEXCEPT {}

    MUDA_GENERIC BufferViewBase(T* data, size_t offset, size_t size) MUDA_NOEXCEPT
        : m_data(data),
          m_offset(offset),
          m_size(size)
    {
    }

    MUDA_GENERIC BufferViewBase(T* data, size_t size) MUDA_NOEXCEPT
        : BufferViewBase(data, 0, size)
    {
    }

    MUDA_GENERIC size_t   size() const MUDA_NOEXCEPT { return m_size; }
    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT
    {
        return m_data + m_offset;
    }

    MUDA_GENERIC const T* data(size_t x) const MUDA_NOEXCEPT
    {
        x += m_offset;
        return m_data + x;
    }

    MUDA_GENERIC const T* origin_data() const MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC size_t   offset() const MUDA_NOEXCEPT { return m_offset; }

    MUDA_GENERIC BufferViewBase subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT;
    MUDA_GENERIC CDense1D<T> cviewer() const MUDA_NOEXCEPT;
};

template <typename T>
class BufferView;

template <typename T>
class CBufferView : public BufferViewBase<T>
{
    using Base = BufferViewBase<T>;

  public:
    using Base::Base;

    MUDA_GENERIC CBufferView(const BufferViewBase<T>& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CBufferView(const T* data, size_t offset, size_t size) MUDA_NOEXCEPT
        : Base(const_cast<T*>(data), offset, size)
    {
    }

    MUDA_GENERIC CBufferView(const T* data, size_t size) MUDA_NOEXCEPT
        : Base(const_cast<T*>(data), size)
    {
    }

    MUDA_GENERIC CBufferView(CDense1D<T> viewer) MUDA_NOEXCEPT
        : Base(const_cast<T*>(viewer.data()), 0, (size_t)viewer.total_size())
    {
    }

    MUDA_GENERIC CBufferView subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT
    {
        return CBufferView{Base::subview(offset, size)};
    }

    MUDA_HOST void copy_to(T* host) const;
};

template <typename T>
class BufferView : public BufferViewBase<T>
{
    using Base = BufferViewBase<T>;

  public:
    using Base::BufferViewBase;
    using Base::data;
    using Base::origin_data;

    MUDA_GENERIC BufferView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC BufferView(Dense1D<T> viewer) MUDA_NOEXCEPT
        : Base(viewer.data(), 0, (size_t)viewer.total_size())
    {
    }

    MUDA_GENERIC operator CBufferView<T>() const MUDA_NOEXCEPT
    {
        return CBufferView<T>{*this};
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::data());
    }

    MUDA_GENERIC T* data(size_t x) MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::data(x));
    }

    MUDA_GENERIC T* origin_data() MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::origin_data());
    }

    MUDA_GENERIC BufferView subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT
    {
        return BufferView{Base::subview(offset, size)};
    }

    MUDA_HOST void fill(const T& v);
    MUDA_HOST void copy_from(CBufferView<T> other);
    MUDA_HOST void copy_from(T* host);
    MUDA_HOST void copy_to(T* host) const
    {
        CBufferView<T>{*this}.copy_to(host);
    }

    MUDA_GENERIC Dense1D<T> viewer() const MUDA_NOEXCEPT;
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