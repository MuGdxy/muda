#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cinttypes>
#include <muda/type_traits/type_modifier.h>
#include <muda/viewer/dense/dense_3d.h>
#include <muda/tools/extent.h>
#include <muda/buffer/buffer_info_accessor.h>

namespace muda
{
template <typename T>
class Buffer3DViewBase
{
    friend class BufferLaunch;
    friend class muda::details::buffer::BufferInfoAccessor<Buffer3DViewBase<T>>;

  private:
    MUDA_GENERIC Buffer3DViewBase(T*              data,
                                  size_t          pitch_bytes,
                                  size_t          pitch_bytes_area,
                                  size_t          origin_width,
                                  size_t          origin_height,
                                  const Offset3D& offset,
                                  const Extent3D& extent) MUDA_NOEXCEPT
        : m_data(data),
          m_pitch_bytes(pitch_bytes),
          m_pitch_bytes_area(pitch_bytes_area),
          m_origin_width(origin_width),
          m_origin_height(origin_height),
          m_offset(offset),
          m_extent(extent)
    {
    }

  protected:
    T*     m_data             = nullptr;
    size_t m_pitch_bytes      = ~0;
    size_t m_pitch_bytes_area = ~0;
    size_t m_origin_width     = ~0;
    size_t m_origin_height    = ~0;

    Offset3D m_offset;
    Extent3D m_extent;

  public:
    MUDA_GENERIC Buffer3DViewBase() MUDA_NOEXCEPT {}

    MUDA_GENERIC Buffer3DViewBase(T*              data,
                                  size_t          pitch_bytes,
                                  size_t          pitch_bytes_area,
                                  const Offset3D& offset,
                                  const Extent3D& extent) MUDA_NOEXCEPT
        : Buffer3DViewBase(
              data, pitch_bytes, pitch_bytes_area, extent.width(), extent.height(), offset, extent)
    {
    }

    MUDA_GENERIC auto extent() const MUDA_NOEXCEPT { return m_extent; }
    MUDA_GENERIC const T* data(size_t x, size_t y, size_t z) const MUDA_NOEXCEPT;
    MUDA_GENERIC const T* data(size_t flatten_i) const MUDA_NOEXCEPT;
    MUDA_GENERIC const T* origin_data() const MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC auto     offset() const MUDA_NOEXCEPT { return m_offset; }
    MUDA_GENERIC auto     pitch_bytes() const MUDA_NOEXCEPT
    {
        return m_pitch_bytes;
    }
    MUDA_GENERIC auto pitch_bytes_area() const MUDA_NOEXCEPT
    {
        return m_pitch_bytes_area;
    }
    MUDA_GENERIC size_t total_size() const MUDA_NOEXCEPT;

    MUDA_GENERIC Buffer3DViewBase<T> subview(Offset3D offset, Extent3D extent = {}) const MUDA_NOEXCEPT;
    MUDA_GENERIC CDense3D<T> cviewer() const MUDA_NOEXCEPT;

  protected:
    MUDA_GENERIC cudaPitchedPtr cuda_pitched_ptr() const MUDA_NOEXCEPT
    {
        return make_cudaPitchedPtr(m_data, m_pitch_bytes, m_origin_width * sizeof(T), m_origin_height);
    }
};

template <typename T>
class CBuffer3DView : public Buffer3DViewBase<T>
{
    using Base = Buffer3DViewBase<T>;

  public:
    using Base::Base;

    MUDA_GENERIC CBuffer3DView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CBuffer3DView(const T*        data,
                               size_t          pitch_bytes,
                               size_t          pitch_bytes_area,
                               const Offset3D& offset,
                               const Extent3D& extent) MUDA_NOEXCEPT
        : Base(const_cast<T*>(data), pitch_bytes, pitch_bytes_area, offset, extent)
    {
    }

    MUDA_GENERIC CBuffer3DView<T> subview(Offset3D offset, Extent3D extent = {}) const MUDA_NOEXCEPT
    {
        return CBuffer3DView<T>{Base::subview(offset, extent)};
    }

    MUDA_HOST void copy_to(T* host) const;
};

template <typename T>
class Buffer3DView : public Buffer3DViewBase<T>
{
    using Base = Buffer3DViewBase<T>;

  public:
    using Base::Base;
    using Base::data;
    using Base::origin_data;

    MUDA_GENERIC Buffer3DView(const Base& base)
        : Base(base)
    {
    }
    
    MUDA_GENERIC Buffer3DView(const CBuffer3DView<T>&) = delete;

    MUDA_GENERIC operator CBuffer3DView<T>() const MUDA_NOEXCEPT
    {
        return CBuffer3DView<T>{*this};
    }

    MUDA_GENERIC T* data(size_t x, size_t y, size_t z) MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::data(x, y, z));
    }

    MUDA_GENERIC T* data(size_t flatten_i) MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::data(flatten_i));
    }

    MUDA_GENERIC T* origin_data() MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::origin_data());
    }

    MUDA_GENERIC Buffer3DView<T> subview(Offset3D offset, Extent3D extent = {}) const MUDA_NOEXCEPT
    {
        return Buffer3DView<T>{Base::subview(offset, extent)};
    }

    MUDA_GENERIC Dense3D<T> viewer() const MUDA_NOEXCEPT;

    MUDA_HOST void fill(const T& v);
    MUDA_HOST void copy_from(const Buffer3DView<T>& other);
    MUDA_HOST void copy_from(const T* host);
    MUDA_HOST void copy_to(T* host) const
    {
        CBuffer3DView<T>{*this}.copy_to(host);
    }

    MUDA_GENERIC cudaPitchedPtr cuda_pitched_ptr() const MUDA_NOEXCEPT
    {
        return Base::cuda_pitched_ptr();
    }
};

template <typename T>
struct read_only_viewer<Buffer3DView<T>>
{
    using type = CBuffer3DView<T>;
};

template <typename T>
struct read_write_viewer<CBuffer3DView<T>>
{
    using type = Buffer3DView<T>;
};
}  // namespace muda

#include "details/buffer_3d_view.inl"