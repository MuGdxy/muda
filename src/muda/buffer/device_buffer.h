#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <muda/viewer/dense.h>

namespace muda
{
template <typename T>
class DeviceVector;

template <typename T>
class HostVector;


template <typename T>
class DeviceBufferView
{

    template <typename T>
    friend class DeviceBuffer;
    T*     m_data   = nullptr;
    size_t m_offset = ~0;
    size_t m_size   = ~0;

  public:
    DeviceBufferView(T* data, size_t offset, size_t size)
        : m_data(data)
        , m_offset(offset)
        , m_size(size)
    {
    }

    DeviceBufferView(T* data, size_t size)
        : m_data(data)
        , m_offset(0)
        , m_size(size)
    {
    }

    size_t   size() const { return m_size; }
    T*       data() { return m_data + m_offset; }
    const T* data() const { return m_data + m_offset; }
    T*       origin_data() { return m_data; }
    const T* origin_data() const { return m_data; }
    size_t   offset() const { return m_offset; }

    DeviceBufferView subview(size_t offset = 0, size_t size = ~0) const;

    void fill(const T& v);
    void copy_from(const DeviceBufferView<T>& other);
    void copy_from(T* host);
    void copy_to(T* host) const;

    Dense1D<T>  viewer();
    CDense1D<T> cviewer() const;
};

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

    DeviceBuffer& operator=(const DeviceBuffer<value_type>& other);
    DeviceBuffer& operator=(const DeviceVector<value_type>& other);
    DeviceBuffer& operator=(const HostVector<value_type>& other);
    DeviceBuffer& operator=(const std::vector<value_type>& other);

    void copy_to(T* host) const;
    void copy_to(std::vector<T>& host) const;

    void resize(size_t new_size);
    void resize(size_t new_size, const value_type& value);
    void clear();
    void shrink_to_fit();
    void fill(const T& v);

    Dense1D<T>  viewer();
    CDense1D<T> cviewer() const;

    DeviceBufferView<T> view(size_t offset, size_t size = ~0) const;
    DeviceBufferView<T> view() const;

    ~DeviceBuffer();

    size_t   size() const { return m_size; }
    T*       data() { return m_data; }
    const T* data() const { return m_data; }
};
}  // namespace muda

#include "details/device_buffer.inl"