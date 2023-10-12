namespace muda
{
template <typename T>
DeviceBufferVar<T>::DeviceBufferVar()
{
    Memory().alloc(&m_data, sizeof(value_type)).wait();
};


template <typename T>
Empty DeviceBufferVar<T>::copy_from(const value_type& var)
{
    m_init = true;
    Memory(m_stream).copy(m_data, std::addressof(var), sizeof(value_type), cudaMemcpyHostToDevice);
    return Empty(m_stream);
}
template <typename T>
Empty DeviceBufferVar<T>::copy_to(value_type& var) const
{
    m_init = true;
    Memory(m_stream).copy(std::addressof(var), m_data, sizeof(value_type), cudaMemcpyDeviceToHost);
    return Empty(m_stream);
}
template <typename T>
Empty DeviceBufferVar<T>::copy_from(const DeviceVar<value_type>& var)
{
    m_init = true;
    Memory(m_stream).copy(m_data, var.data(), sizeof(value_type), cudaMemcpyDeviceToDevice);
    return Empty(m_stream);
}
template <typename T>
Empty DeviceBufferVar<T>::copy_to(DeviceVar<value_type>& var) const
{
    m_init = true;
    Memory(m_stream).copy(var.data(), m_data, sizeof(value_type), cudaMemcpyDeviceToDevice);
    return Empty(m_stream);
}
template <typename T>
Empty DeviceBufferVar<T>::copy_from(const DeviceBufferVar<value_type>& var)
{
    m_init = true;
    Memory(m_stream).copy(m_data, var.data(), sizeof(value_type), cudaMemcpyDeviceToDevice);
    return Empty(m_stream);
}
template <typename T>
Empty DeviceBufferVar<T>::copy_to(DeviceBufferVar<value_type>& var) const
{
    m_init = true;
    Memory(m_stream).copy(var.data(), m_data, sizeof(value_type), cudaMemcpyDeviceToDevice);
    return Empty(m_stream);
}
template <typename T>
DeviceBufferVar<T>& DeviceBufferVar<T>::operator=(const DeviceBufferVar<value_type>& other)
{
    if(&other == this)
        return *this;
    copy_from(other).wait();
    return *this;
}
template <typename T>
DeviceBufferVar<T>& DeviceBufferVar<T>::operator=(const DeviceVar<value_type>& other)
{
    copy_from(other).wait();
    return *this;
}
template <typename T>
DeviceBufferVar<T>& DeviceBufferVar<T>::operator=(const value_type& other)
{
    copy_from(other).wait();
    return *this;
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