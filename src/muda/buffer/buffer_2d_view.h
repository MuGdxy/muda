#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cinttypes>
#include <muda/type_traits/type_modifier.h>
#include <muda/viewer/dense/dense_2d.h>
#include <muda/tools/extent.h>
#include <muda/buffer/buffer_info_accessor.h>

namespace muda
{
template <typename T>
class Buffer2DViewBase
{
    friend class BufferLaunch;
    friend class muda::details::buffer::BufferInfoAccessor<Buffer2DViewBase<T>>;

  private:
    MUDA_GENERIC Buffer2DViewBase(T*              data,
                                  size_t          pitch_bytes,
                                  size_t          origin_width,
                                  size_t          origin_height,
                                  const Offset2D& offset,
                                  const Extent2D& extent) MUDA_NOEXCEPT
        : m_data(data),
          m_pitch_bytes(pitch_bytes),
          m_origin_width(origin_width),
          m_origin_height(origin_height),
          m_offset(offset),
          m_extent(extent)
    {
    }

  protected:
    T*       m_data          = nullptr;
    size_t   m_pitch_bytes   = ~0;
    size_t   m_origin_width  = 0;
    size_t   m_origin_height = 0;
    Offset2D m_offset;
    Extent2D m_extent;

  public:
    MUDA_GENERIC Buffer2DViewBase() MUDA_NOEXCEPT {}

    MUDA_GENERIC Buffer2DViewBase(T*              data,
                                  size_t          pitch_bytes,
                                  const Offset2D& offset,
                                  const Extent2D& extent) MUDA_NOEXCEPT
        : Buffer2DViewBase(data, pitch_bytes, extent.width(), extent.height(), offset, extent)
    {
    }

    MUDA_GENERIC auto   extent() const MUDA_NOEXCEPT { return m_extent; }
    MUDA_GENERIC size_t pitch_bytes() const MUDA_NOEXCEPT
    {
        return m_pitch_bytes;
    }
    MUDA_GENERIC const T* data(size_t x, size_t y) const MUDA_NOEXCEPT;
    MUDA_GENERIC const T* data(size_t flatten_i) const MUDA_NOEXCEPT;
    MUDA_GENERIC const T* origin_data() const MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC auto     offset() const MUDA_NOEXCEPT { return m_offset; }
    MUDA_GENERIC auto     total_size() const MUDA_NOEXCEPT
    {
        return m_extent.width() * m_extent.height();
    }

    MUDA_GENERIC Buffer2DViewBase<T> subview(Offset2D offset, Extent2D extent = {}) const MUDA_NOEXCEPT;
    MUDA_GENERIC CDense2D<T> cviewer() const MUDA_NOEXCEPT;

  protected:
    MUDA_GENERIC cudaPitchedPtr cuda_pitched_ptr() const MUDA_NOEXCEPT;
};

template <typename T>
class CBuffer2DView : public Buffer2DViewBase<T>
{
    using Base = Buffer2DViewBase<T>;

  public:
    using Base::Base;

    MUDA_GENERIC CBuffer2DView(const Base& base) MUDA_NOEXCEPT : Base(base) {}

    MUDA_GENERIC CBuffer2DView(const T*        data,
                               size_t          pitch_bytes,
                               const Offset2D& offset,
                               const Extent2D& extent) MUDA_NOEXCEPT
        : Base(const_cast<T*>(data), pitch_bytes, offset, extent)
    {
    }

    MUDA_GENERIC CBuffer2DView<T> subview(Offset2D offset, Extent2D extent = {}) const MUDA_NOEXCEPT
    {
        return CBuffer2DView<T>{Base::subview(offset, extent)};
    }

    MUDA_HOST void copy_to(T* host) const;
};

template <typename T>
class Buffer2DView : public Buffer2DViewBase<T>
{
    using Base = Buffer2DViewBase<T>;

  public:
    using Base::Base;
    using Base::data;
    using Base::origin_data;

    MUDA_GENERIC Buffer2DView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC Buffer2DView(const CBuffer2DView<T>&) = delete;

    MUDA_GENERIC operator CBuffer2DView<T>() const MUDA_NOEXCEPT
    {
        return CBuffer2DView<T>{*this};
    }
    MUDA_GENERIC T* data(size_t x, size_t y) const MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::data(x, y));
    }
    MUDA_GENERIC T* data(size_t flatten_i) const MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::data(flatten_i));
    }
    MUDA_GENERIC T* origin_data() MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::origin_data());
    }

    MUDA_GENERIC Buffer2DView<T> subview(Offset2D offset, Extent2D extent = {}) MUDA_NOEXCEPT
    {
        return Buffer2DView<T>{Base::subview(offset, extent)};
    }

    MUDA_GENERIC CBuffer2DView<T> subview(Offset2D offset, Extent2D extent = {}) const MUDA_NOEXCEPT
    {
        return CBuffer2DView<T>{Base::subview(offset, extent)};
    }

    MUDA_GENERIC Dense2D<T> viewer() const MUDA_NOEXCEPT;

    MUDA_HOST void fill(const T& v);
    MUDA_HOST void copy_from(CBuffer2DView<T> other);
    MUDA_HOST void copy_from(const T* host);
    MUDA_HOST void copy_to(T* host) const
    {
        return CBuffer2DView<T>{*this}.copy_to(host);
    }

    MUDA_GENERIC cudaPitchedPtr cuda_pitched_ptr() const MUDA_NOEXCEPT
    {
        return Base::cuda_pitched_ptr();
    }
};

template <typename T>
struct read_only_viewer<Buffer2DView<T>>
{
    using type = CBuffer2DView<T>;
};

template <typename T>
struct read_write_viewer<CBuffer2DView<T>>
{
    using type = Buffer2DView<T>;
};
}  // namespace muda

#include "details/buffer_2d_view.inl"