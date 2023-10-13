#include <muda/buffer/buffer_launch.h>
#include <muda/container/vector.h>
#include <muda/container/var.h>
#include <muda/launch/memory.h>
#include <muda/launch/parallel_for.h>

namespace muda
{
template <typename T>
DeviceBufferView<T> DeviceBufferView<T>::subview(size_t offset, size_t size) const
{
    if(size == ~0)
        size = m_size - offset;
    MUDA_ASSERT(offset + size <= m_size,
                "DeviceBufferView out of range, size = %d, yours = %d",
                m_size,
                offset + size);
    return DeviceBufferView(m_data, m_offset + offset, size);
}

template <typename T>
void DeviceBufferView<T>::fill(const T& v)
{
    BufferLaunch()
        .fill(*this, v)  //
        .wait();
}

template <typename T>
void DeviceBufferView<T>::copy_from(const DeviceBufferView<T>& other)
{
    BufferLaunch()
        .copy(*this, other)  //
        .wait();
}

template <typename T>
void DeviceBufferView<T>::copy_from(T* host)
{
    BufferLaunch()
        .copy(*this, host)  //
        .wait();
}

template <typename T>
void DeviceBufferView<T>::copy_to(T* host) const
{
    BufferLaunch()
        .copy(host, *this)  //
        .wait();
}

template <typename T>
Dense1D<T> DeviceBufferView<T>::viewer()
{
    return Dense1D<T>(m_data, m_size);
}

template <typename T>
CDense1D<T> DeviceBufferView<T>::cviewer() const
{
    return CDense1D<T>(m_data, m_size);
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(size_t n)
{
    BufferLaunch()
        .alloc(*this, n)  //
        .wait();
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer()
    : m_data(nullptr)
    , m_size(0)
    , m_capacity(0)
{
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer<T>&& other) MUDA_NOEXCEPT
    : m_data(other.m_data),
      m_size(other.m_size),
      m_capacity(other.m_capacity)
{
    other.m_data = nullptr;
    other.m_size = 0;
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const DeviceBuffer<T>& other)
{
    BufferLaunch()
        .alloc(*this, other.size())  //
        .copy(view(), other.view())  //
        .wait();
}


template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(const DeviceBuffer<value_type>& other)
{
    if(this == &other)
        return *this;
    BufferLaunch()
        .resize(*this, other.size())  //
        .copy(view(), other.view())   //
        .wait();
    return *this;
}

template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(const DeviceVector<value_type>& other)
{
    BufferLaunch()
        .resize(*this, other.size())                                    //
        .copy(view(), DeviceBufferView{other.data(), 0, other.size()})  //
        .wait();
    return *this;
}

template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(const HostVector<value_type>& other)
{
    BufferLaunch()
        .resize(*this, other.size())  //
        .copy(view(), other.data())   //
        .wait();
    return *this;
}

template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(const std::vector<value_type>& other)
{
    BufferLaunch()
        .resize(*this, other.size())  //
        .copy(view(), other.data())   //
        .wait();
    return *this;
}

template <typename T>
void DeviceBuffer<T>::copy_to(T* host) const
{
    view().copy_to(host);
}

template <typename T>
void DeviceBuffer<T>::copy_to(std::vector<T>& host) const
{
    host.resize(size());
    view().copy_to(host.data());
}


template <typename T>
void DeviceBuffer<T>::resize(size_t new_size)
{
    BufferLaunch()
        .resize(*this, new_size)  //
        .wait();
}

template <typename T>
void DeviceBuffer<T>::resize(size_t new_size, const value_type& value)
{
    BufferLaunch()
        .resize(*this, new_size, value)  //
        .wait();
}

template <typename T>
void DeviceBuffer<T>::clear()
{
    BufferLaunch()
        .clear(*this)  //
        .wait();
}

template <typename T>
void DeviceBuffer<T>::shrink_to_fit()
{
    BufferLaunch()
        .shrink_to_fit(*this)  //
        .wait();
}

template <typename T>
void DeviceBuffer<T>::fill(const T& v)
{
    BufferLaunch()
        .fill(view(), v)  //
        .wait();
};

template <typename T>
Dense1D<T> DeviceBuffer<T>::viewer()
{
    return view().viewer();
}

template <typename T>
CDense1D<T> DeviceBuffer<T>::cviewer() const
{
    return view().cviewer();
}

template <typename T>
DeviceBufferView<T> DeviceBuffer<T>::view(size_t offset, size_t size) const
{
    return view().subview(offset, size);
}

template <typename T>
DeviceBufferView<T> DeviceBuffer<T>::view() const
{
    return DeviceBufferView(m_data, 0, m_size);
}

template <typename T>
DeviceBuffer<T>::~DeviceBuffer()
{
    if(m_data)
    {
        BufferLaunch()
            .free(*this)  //
            .wait();
    }
}
}  // namespace muda
