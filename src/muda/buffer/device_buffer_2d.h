#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <muda/viewer/dense.h>
#include <muda/buffer/buffer_2d_view.h>

namespace muda
{
template <typename T>
class DeviceBuffer2D
{
  private:
    friend class BufferLaunch;
    T*       m_data        = nullptr;
    size_t   m_pitch_bytes = 0;
    Extent2D m_extent      = Extent2D::Zero();
    Extent2D m_capacity    = Extent2D::Zero();

  public:
    using value_type = T;

    DeviceBuffer2D(const Extent2D& n);
    DeviceBuffer2D();
    DeviceBuffer2D(const DeviceBuffer2D<T>& other);
    DeviceBuffer2D(const std::vector<T>& host);
    DeviceBuffer2D(DeviceBuffer2D&& other) MUDA_NOEXCEPT;

    DeviceBuffer2D& operator=(Buffer2DView<T> view);
    DeviceBuffer2D& operator=(const DeviceBuffer2D<T>& other);
    DeviceBuffer2D& operator=(const std::vector<T>& other);

    void copy_to(T* host) const;
    void copy_to(std::vector<T>& host) const;
    void copy_from(const T* host);
    void copy_from(const std::vector<T>& host);

    void resize(size_t new_size);
    void resize(size_t new_size, const T& value);
    void clear();
    void shrink_to_fit();
    void fill(const T& v);

    Dense3D<T>  viewer() MUDA_NOEXCEPT { return view().viewer(); }
    CDense3D<T> cviewer() const MUDA_NOEXCEPT { return view().viewer(); }

    Buffer2DView<T> view(Offset2D offset, Extent2D extent = {}) MUDA_NOEXCEPT
    {
        return Buffer2DView<T>{m_data, m_pitch_bytes, m_extent, offset, extent};
    }
    Buffer2DView<T> view() MUDA_NOEXCEPT
    {
        return Buffer2DView<T>{m_data, m_pitch_bytes, Offset2D::Zero(), m_extent};
    }
    operator Buffer2DView<T>() MUDA_NOEXCEPT { return view(); }

    CBuffer2DView<T> view(Offset2D offset, Extent2D extent = {}) const MUDA_NOEXCEPT
    {
        return CBuffer2DView<T>{m_data, m_pitch_bytes, m_extent, offset, extent};
    }

    CBuffer2DView<T> view() const MUDA_NOEXCEPT
    {
        return CBuffer2DView<T>{m_data, m_pitch_bytes, Offset2D::Zero(), m_extent};
    }
    operator CBuffer2DView<T>() const MUDA_NOEXCEPT { return view(); }

    ~DeviceBuffer2D();

    auto     extent() const MUDA_NOEXCEPT { return m_extent; }
    auto     capacity() const MUDA_NOEXCEPT { return m_capacity; }
    T*       data() MUDA_NOEXCEPT { return m_data; }
    const T* data() const MUDA_NOEXCEPT { return m_data; }
};
}  // namespace muda