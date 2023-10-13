#include <muda/buffer/buffer_launch.h>

namespace muda
{
template <typename T>
DeviceBufferVar<T>::DeviceBufferVar()
{
    Memory().alloc(&m_data, sizeof(value_type)).wait();
};

template <typename T>
DeviceBufferVar<T>::DeviceBufferVar(const DeviceBufferVar& other)
{
    Memory().alloc(&m_data, sizeof(value_type)).wait();
    BufferLaunch()
        .copy(*this, other)  //
        .wait();
}

template <typename T>
DeviceBufferVar<T>& DeviceBufferVar<T>::operator=(const DeviceBufferVar<value_type>& other)
{
    if(this == &other)
        return *this;

    BufferLaunch()
        .copy(*this, other)  //
        .wait();
    return *this;
}

template <typename T>
DeviceBufferVar<T>& DeviceBufferVar<T>::operator=(const DeviceVar<value_type>& other)
{
    BufferLaunch()
        .copy(*this, other)  //
        .wait();
    return *this;
}

template <typename T>
DeviceBufferVar<T>& DeviceBufferVar<T>::operator=(const value_type& val)
{
    BufferLaunch()
        .fill(*this, val)  //
        .wait();
    return *this;
}

template <typename T>
DeviceBufferVar<T>::DeviceBufferVar(DeviceBufferVar&& other) MUDA_NOEXCEPT
    : m_data(other.m_data)
{
    other.m_data = nullptr;
}

template <typename T>
DeviceBufferVar<T>::operator T() const
{
    T var;
    BufferLaunch()
        .copy(&var, *this)  //
        .wait();
    return var;
}

template <typename T>
Dense<T> DeviceBufferVar<T>::viewer()
{
    return Dense<T>(m_data);
}
template <typename T>
CDense<T> DeviceBufferVar<T>::cviewer() const
{
    return CDense<T>(m_data);
}
}  // namespace muda
