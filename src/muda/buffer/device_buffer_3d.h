#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <muda/viewer/dense.h>
#include <muda/buffer/buffer_3d_view.h>

namespace muda
{
template <typename T>
class DeviceBuffer3D
{
  private:
    friend class BufferLaunch;
    T*       m_data             = nullptr;
    size_t   m_pitch_bytes      = 0;
    size_t   m_pitch_bytes_area = 0;
    Extent3D m_extent;
    Extent3D m_capacity;

  public:
    using value_type = T;

    DeviceBuffer3D(const Extent3D& n);
    DeviceBuffer3D();
    DeviceBuffer3D(const DeviceBuffer3D<T>& other);
    DeviceBuffer3D(const std::vector<T>& host);
    DeviceBuffer3D(DeviceBuffer3D&& other) MUDA_NOEXCEPT;

    DeviceBuffer3D& operator=(Buffer3DView<T> view);
    DeviceBuffer3D& operator=(const DeviceBuffer3D<T>& other);
    DeviceBuffer3D& operator=(const std::vector<T>& other);

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

    Buffer3DView<T> view(Offset3D offset, Extent3D extent = {}) MUDA_NOEXCEPT
    {
        return Buffer3DView<T>{m_data, m_pitch_bytes, m_pitch_bytes_area, m_extent, offset, extent};
    }
    Buffer3DView<T> view() MUDA_NOEXCEPT
    {
        return Buffer3DView<T>{
            m_data, m_pitch_bytes, m_pitch_bytes_area, Offset3D::Zero(), m_extent};
    }
    operator Buffer3DView<T>() MUDA_NOEXCEPT { return view(); }

    CBuffer3DView<T> view(Offset3D offset, Extent3D extent = {}) const MUDA_NOEXCEPT
    {
        return CBuffer3DView<T>{m_data, m_pitch_bytes, m_pitch_bytes_area, m_extent, offset, extent};
    }

    CBuffer3DView<T> view() const MUDA_NOEXCEPT
    {
        return CBuffer3DView<T>{
            m_data, m_pitch_bytes, m_pitch_bytes_area, Offset3D::Zero(), m_extent};
    }
    operator CBuffer3DView<T>() const MUDA_NOEXCEPT { return view(); }

    ~DeviceBuffer3D();

    auto     extent() const MUDA_NOEXCEPT { return m_extent; }
    auto     capacity() const MUDA_NOEXCEPT { return m_capacity; }
    T*       data() MUDA_NOEXCEPT { return m_data; }
    const T* data() const MUDA_NOEXCEPT { return m_data; }
};
}  // namespace muda