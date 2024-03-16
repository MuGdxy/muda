#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cinttypes>
#include <muda/type_traits/type_modifier.h>
#include <muda/viewer/dense/dense_3d.h>
#include <muda/tools/extent.h>
#include <muda/buffer/buffer_info_accessor.h>
#include <muda/view/view_base.h>
namespace muda
{
template <bool IsConst, typename T>
class Buffer3DViewBase : public ViewBase<IsConst>
{
    using Base = ViewBase<IsConst>;
  public:
    static_assert(!std::is_const_v<T>, "Ty must be non-const");
    using ConstView    = Buffer3DViewBase<true, T>;
    using NonConstView = Buffer3DViewBase<false, T>;
    using ThisView     = Buffer3DViewBase<IsConst, T>;
    using OtherView    = Buffer3DViewBase<!IsConst, T>;

    using CViewer    = CDense3D<T>;
    using Viewer     = Dense3D<T>;
    using ThisViewer = std::conditional_t<IsConst, CViewer, Viewer>;

  private:
    friend class BufferLaunch;
    friend class details::buffer::BufferInfoAccessor<ThisView>;

    template<typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  protected:
    auto_const_t<T>* m_data             = nullptr;
    size_t           m_pitch_bytes      = ~0;
    size_t           m_pitch_bytes_area = ~0;
    size_t           m_origin_width     = ~0;
    size_t           m_origin_height    = ~0;

    Offset3D m_offset;
    Extent3D m_extent;

  public:
    MUDA_GENERIC Buffer3DViewBase() MUDA_NOEXCEPT = default;

    MUDA_GENERIC Buffer3DViewBase(auto_const_t<T>* data,
                                  size_t           pitch_bytes,
                                  size_t           pitch_bytes_area,
                                  size_t           origin_width,
                                  size_t           origin_height,
                                  const Offset3D&  offset,
                                  const Extent3D&  extent) MUDA_NOEXCEPT
        : m_data(data),
          m_pitch_bytes(pitch_bytes),
          m_pitch_bytes_area(pitch_bytes_area),
          m_origin_width(origin_width),
          m_origin_height(origin_height),
          m_offset(offset),
          m_extent(extent)
    {
    }

    MUDA_GENERIC Buffer3DViewBase(T*              data,
                                  size_t          pitch_bytes,
                                  size_t          pitch_bytes_area,
                                  const Offset3D& offset,
                                  const Extent3D& extent) MUDA_NOEXCEPT
        : Buffer3DViewBase(
              data, pitch_bytes, pitch_bytes_area, extent.width(), extent.height(), offset, extent)
    {
    }

    // implicit conversion

    ConstView as_const() const MUDA_NOEXCEPT
    {
        return ConstView{m_data, m_pitch_bytes, m_pitch_bytes_area, m_offset, m_extent};
    }

    operator ConstView() const MUDA_NOEXCEPT { return as_const(); }

    // non-const accessor
    MUDA_GENERIC auto_const_t<T>* data(size_t x, size_t y, size_t z) MUDA_NOEXCEPT;
    MUDA_GENERIC auto_const_t<T>* data(size_t flatten_i) MUDA_NOEXCEPT;
    MUDA_GENERIC auto_const_t<T>* origin_data() MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC ThisView subview(Offset3D offset, Extent3D extent = {}) MUDA_NOEXCEPT;
    MUDA_GENERIC ThisViewer viewer() MUDA_NOEXCEPT;

    // const accessor

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
    MUDA_GENERIC ConstView subview(Offset3D offset, Extent3D extent = {}) const MUDA_NOEXCEPT;
    MUDA_GENERIC CViewer        cviewer() const MUDA_NOEXCEPT;
    MUDA_GENERIC cudaPitchedPtr cuda_pitched_ptr() const MUDA_NOEXCEPT
    {
        return make_cudaPitchedPtr(remove_const(m_data),
                                   remove_const(m_pitch_bytes),
                                   m_origin_width * sizeof(T),
                                   m_origin_height);
    }
};

template <typename T>
class CBuffer3DView : public Buffer3DViewBase<true, T>
{
    using Base = Buffer3DViewBase<true, T>;

  public:
    using Base::Base;

    MUDA_GENERIC CBuffer3DView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CBuffer3DView<T> subview(Offset3D offset, Extent3D extent = {}) const MUDA_NOEXCEPT
    {
        return CBuffer3DView<T>{Base::subview(offset, extent)};
    }

    MUDA_HOST void copy_to(T* host) const;

    MUDA_GENERIC auto as_const() const MUDA_NOEXCEPT { return *this; }
};

template <typename T>
class Buffer3DView : public Buffer3DViewBase<false, T>
{
    using Base = Buffer3DViewBase<false, T>;

  public:
    using Base::Base;

    MUDA_GENERIC Buffer3DView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC Buffer3DView(const CBuffer3DView<T>&) = delete;

    MUDA_GENERIC CBuffer3DView<T> as_const() const MUDA_NOEXCEPT
    {
        return CBuffer3DView<T>{Base::as_const()};
    }

    MUDA_GENERIC operator CBuffer3DView<T>() const MUDA_NOEXCEPT
    {
        return as_const();
    }

    MUDA_HOST void fill(const T& v);
    MUDA_HOST void copy_from(const Buffer3DView<T>& other);
    MUDA_HOST void copy_from(const T* host);
    MUDA_HOST void copy_to(T* host) const
    {
        CBuffer3DView<T>{*this}.copy_to(host);
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