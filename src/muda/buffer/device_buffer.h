#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <muda/viewer/dense.h>
#include <muda/buffer/buffer_view.h>

namespace muda
{
template <typename T>
class DeviceVector;

template <typename T>
class HostVector;

template <typename T>
class DeviceBuffer
{
  private:
    friend class BufferLaunch;
    size_t m_size     = 0;
    size_t m_capacity = 0;
    T*     m_data     = nullptr;

  public:
    using value_type = T;

    DeviceBuffer(size_t n);
    DeviceBuffer();
    DeviceBuffer(const DeviceBuffer<T>& other);
    DeviceBuffer(DeviceBuffer&& other) MUDA_NOEXCEPT;

    DeviceBuffer& operator=(BufferView<T> view);
    DeviceBuffer& operator=(const DeviceBuffer<T>& other);
    DeviceBuffer& operator=(const std::vector<T>& other);

    void copy_to(T* host) const;
    void copy_to(std::vector<T>& host) const;

    void resize(size_t new_size);
    void resize(size_t new_size, const T& value);
    void clear();
    void shrink_to_fit();
    void fill(const T& v);

    Dense1D<T>  viewer() MUDA_NOEXCEPT;
    CDense1D<T> cviewer() const MUDA_NOEXCEPT;

    BufferView<T> view(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT;
    BufferView<T> view() const MUDA_NOEXCEPT;
    operator BufferView<T>() const MUDA_NOEXCEPT { return view(); }

    ~DeviceBuffer();

    size_t   size() const MUDA_NOEXCEPT { return m_size; }
    T*       data() MUDA_NOEXCEPT { return m_data; }
    const T* data() const MUDA_NOEXCEPT { return m_data; }
};
}  // namespace muda

namespace muda
{
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense(DeviceBuffer<T>& v) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense(const DeviceBuffer<T>& v) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_viewer(DeviceBuffer<T>& v) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cviewer(const DeviceBuffer<T>& v) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(DeviceBuffer<T>& v, int dimy) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense2D(const DeviceBuffer<T>& v, int dimy) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(DeviceBuffer<T>& v, int dimx, int dimy) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense2D(const DeviceBuffer<T>& v, int dimx, int dimy) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(DeviceBuffer<T>& v, const int2& dim) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense2D(const DeviceBuffer<T>& v, const int2& dim) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(DeviceBuffer<T>& v, int dimy, int dimz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(const DeviceBuffer<T>& v, int dimy, int dimz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(DeviceBuffer<T>& v, const int2& dimyz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(const DeviceBuffer<T>& v, const int2& dimyz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(DeviceBuffer<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(const DeviceBuffer<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(DeviceBuffer<T>& v, const int3& dim) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(const DeviceBuffer<T>& v, const int3& dim) MUDA_NOEXCEPT;
}

#include "details/device_buffer.inl"