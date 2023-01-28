#ifdef __INTELLISENSE__
#include "device_buffer.h"
#endif

namespace muda
{
template <typename T>
empty device_buffer<T>::resize(size_t new_size, buf_op mem_op, char setbyte)
{
    auto   mem      = memory(m_stream);
    size_t old_size = m_size;

    if(new_size <= m_size)
    {
        switch(mem_op)
        {
            case muda::buf_op::set:
                mem.set(m_data, new_size * sizeof(value_type), (int)setbyte);
                break;
            default:
                break;
        }
        m_size = new_size;
        return empty(m_stream);
    }

    if(new_size <= m_capacity)
    {
        switch(mem_op)
        {
            case muda::buf_op::set:
                mem.set(m_data, new_size * sizeof(value_type), (int)setbyte);
                break;
            case muda::buf_op::keep_set:
                mem.set(m_data + old_size,
                        (new_size - old_size) * sizeof(value_type),
                        (int)setbyte);
                break;
            default:
                break;
        }
        m_size = new_size;
    }
    else
    {
        T* ptr;
        mem.alloc(&ptr, new_size * sizeof(value_type));
        switch(mem_op)
        {
            case muda::buf_op::keep:
                mem.copy(ptr, m_data, old_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
                break;
            case muda::buf_op::set:
                mem.set(ptr, new_size * sizeof(value_type), (int)setbyte);
                break;
            case muda::buf_op::keep_set:
                if(m_data)
                    mem.copy(ptr, m_data, old_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
                mem.set(ptr + old_size, (new_size - old_size) * sizeof(value_type), (int)setbyte);
                break;
            default:
                break;
        }
        if(m_data)
            mem.free(m_data);
        m_data     = ptr;
        m_size     = new_size;
        m_capacity = new_size;
    }

    return empty(m_stream);
}

template <typename T>
empty device_buffer<T>::resize(size_t new_size)
{
    auto   mem      = memory(m_stream);
    size_t old_size = m_size;

    if(new_size <= m_size)
    {
        m_size = new_size;
        return empty(m_stream);
    }

    if(new_size <= m_capacity)
    {
        mem.set(m_data + old_size, (new_size - old_size) * sizeof(value_type), 0);
        m_size = new_size;
    }
    else
    {
        T* ptr;
        mem.alloc(&ptr, new_size * sizeof(value_type));
        if(m_data)
            mem.copy(ptr, m_data, old_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        mem.set(ptr + old_size, (new_size - old_size) * sizeof(value_type), 0);
        if(m_data)
            mem.free(m_data);
        m_data     = ptr;
        m_size     = new_size;
        m_capacity = new_size;
    }

    return empty(m_stream);
}

template <typename T>
inline empty device_buffer<T>::resize(size_t new_size, const value_type& value, int blockDim)
{
    auto   mem      = memory(m_stream);
    size_t old_size = m_size;

    if(new_size <= m_capacity)
    {
        m_size = new_size;
    }
    else
    {
        T* ptr;
        mem.alloc(&ptr, new_size * sizeof(value_type));
        if(m_data)
            mem.free(m_data);
        m_data     = ptr;
        m_size     = new_size;
        m_capacity = new_size;
    }

    parallel_for(blockDim, 0, m_stream)
        .apply(new_size,
               [=, d = make_viewer(*this)] __device__(int i) mutable
               { d(i) = value; });

    return empty(m_stream);
}

template <typename T>
empty device_buffer<T>::shrink_to_fit()
{
    auto mem = memory(m_stream);

    if(m_size < m_capacity)
    {
        T* ptr;
        mem.alloc(&ptr, m_size * sizeof(value_type));
        mem.copy(ptr, m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if(m_data)
            mem.free(m_data);
        m_data     = ptr;
        m_capacity = m_size;
    }

    return empty(m_stream);
}

template <typename T>
empty device_buffer<T>::set(char setbyte, size_t count)
{
    m_init = true;
    if(count == size_t(-1))
        count = m_size;
    if(count > m_size)
        throw std::out_of_range("device_buffer::set out of range");
    memory(m_stream).set(m_data, count * sizeof(T), setbyte);
    return empty(m_stream);
}

template <typename T>
empty device_buffer<T>::copy_to(value_type& var) const
{
    if(m_size != 1)
        throw std::logic_error("buffer size larger than 1, cannot copy to host_var");
    m_init = true;
    memory(m_stream).copy(std::addressof(var), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToHost);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_to(host_vector<value_type>& vec) const
{
    m_init = true;
    vec.resize(m_size);
    memory(m_stream).copy(muda::data(vec), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToHost);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_to(device_var<value_type>& var) const
{
    if(m_size != 1)
        throw std::logic_error("buffer size larger than 1, cannot copy to device_var");
    m_init = true;
    memory(m_stream).copy(muda::data(var), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_to(device_vector<value_type>& vec) const
{
    m_init = true;
    vec.resize(m_size);
    memory(m_stream).copy(muda::data(vec), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_to(device_buffer<value_type>& vec) const
{
    m_init = true;
    vec.resize(m_size);
    memory(m_stream).copy(vec.data(), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_from(const host_var<value_type>& var)
{
    m_init = true;
    resize(1);
    memory(m_stream).copy(m_data, muda::data(var), m_size * sizeof(value_type), cudaMemcpyHostToDevice);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_from(const value_type& var)
{
    m_init = true;
    resize(1);
    memory(m_stream).copy(m_data, std::addressof(var), m_size * sizeof(value_type), cudaMemcpyHostToDevice);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_from(const host_vector<value_type>& vec)
{
    m_init = true;
    resize(vec.size());
    memory(m_stream).copy(m_data, muda::data(vec), m_size * sizeof(value_type), cudaMemcpyHostToDevice);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_from(const device_var<value_type>& var)
{
    m_init = true;
    resize(1);
    memory(m_stream).copy(m_data, muda::data(var), m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_from(const device_vector<value_type>& vec)
{
    m_init = true;
    resize(vec.size());
    memory().copy(m_data, muda::data(vec), m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
    return empty(m_stream);
}
template <typename T>
empty device_buffer<T>::copy_from(const device_buffer<value_type>& vec)
{
    m_init = true;
    resize(vec.size());
    memory().copy(m_data, vec.data(), m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
    return empty(m_stream);
}
}