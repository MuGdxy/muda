#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <muda/viewer/dense.h>
#include <muda/buffer/buffer_view.h>

namespace muda
{
class NDReshaper;

template <typename T>
class DeviceVector;

template <typename T>
class HostVector;

template <typename T>
class DeviceBuffer
{
  private:
    friend class BufferLaunch;
    friend class NDReshaper;

    size_t m_size     = 0;
    size_t m_capacity = 0;
    T*     m_data     = nullptr;

  public:
    using value_type = T;

    DeviceBuffer(size_t n);
    DeviceBuffer();

    DeviceBuffer(const DeviceBuffer<T>& other);
    DeviceBuffer(DeviceBuffer&& other) MUDA_NOEXCEPT;
    DeviceBuffer& operator=(const DeviceBuffer<T>& other);
    DeviceBuffer& operator=(DeviceBuffer<T>&& other);

    DeviceBuffer(CBufferView<T> other);
    DeviceBuffer(const std::vector<T>& host);
    DeviceBuffer& operator=(CBufferView<T> other);
    DeviceBuffer& operator=(const std::vector<T>& other);

    void copy_to(std::vector<T>& host) const;
    void copy_from(const std::vector<T>& host);

    void resize(size_t new_size);
    void resize(size_t new_size, const T& value);
    void reserve(size_t new_capacity);
    void clear();
    void shrink_to_fit();
    void fill(const T& v);

    Dense1D<T>  viewer() MUDA_NOEXCEPT;
    CDense1D<T> cviewer() const MUDA_NOEXCEPT;

    BufferView<T>  view(size_t offset, size_t size = ~0) MUDA_NOEXCEPT;
    BufferView<T>  view() MUDA_NOEXCEPT;
    CBufferView<T> view(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT;
    CBufferView<T> view() const MUDA_NOEXCEPT;
    operator BufferView<T>() MUDA_NOEXCEPT { return view(); }
    operator CBufferView<T>() const MUDA_NOEXCEPT { return view(); }

    ~DeviceBuffer();

    auto     size() const MUDA_NOEXCEPT { return m_size; }
    auto     capacity() const MUDA_NOEXCEPT { return m_capacity; }
    T*       data() MUDA_NOEXCEPT { return m_data; }
    const T* data() const MUDA_NOEXCEPT { return m_data; }
};
}  // namespace muda

#include "details/device_buffer.inl"