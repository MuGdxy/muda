#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_buffer_var.h>

namespace muda::details
{
template <typename T>
void kernel_destruct(int grid_dim, int block_dim, cudaStream_t stream, T* data, size_t size)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(size, [d = data] __device__(int i) mutable { d[i].~T(); });
}

template <typename T>
void kernel_construct(int grid_dim, int block_dim, cudaStream_t stream, T* data, size_t size)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(size, [d = data] __device__(int i) mutable { new(d + i) T(); });
}

template <typename T>
void kernel_assign(int grid_dim, int block_dim, cudaStream_t stream, T* dst, const T* src, size_t size)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(size, [dst, src] __device__(int i) mutable { dst[i] = src[i]; });
}

template <typename T>
void kernel_fill(int grid_dim, int block_dim, cudaStream_t stream, T* dst, const T& val, size_t size)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(size, [dst, val] __device__(int i) mutable { dst[i] = val; });
}
}  // namespace muda::details


namespace muda
{
template <typename T>
BufferLaunch& BufferLaunch::resize(DeviceBuffer<T>& buffer, size_t new_size)
{
    auto old_size = buffer.m_size;
    return resize(
        buffer,
        new_size,
        [&](T* ptr)  // construct
        {
            if constexpr(std::is_trivially_constructible_v<T>)
            {
                Memory(m_stream).set(ptr + old_size, (new_size - old_size) * sizeof(T), 0);
            }
            else
            {
                static_assert(std::is_constructible_v<T>,
                              "The type T must be constructible, which means T must have a 0-arg constructor");

                details::kernel_construct(
                    m_grid_dim, m_block_dim, m_stream, ptr + old_size, new_size - old_size);
            }
        });
}

template <typename T>
BufferLaunch& BufferLaunch::resize(DeviceBuffer<T>& buffer, size_t new_size, const T& value)
{
    auto old_size = buffer.m_size;
    return resize(buffer,
                  new_size,
                  [&](T* ptr)
                  {
                      ParallelFor(m_grid_dim, m_block_dim, 0, m_stream)
                          .apply(new_size - old_size,
                                 [value, d = ptr + old_size] __device__(int i) mutable
                                 { d[i] = value; });
                  });
}

template <typename T>
BufferLaunch& BufferLaunch::clear(DeviceBuffer<T>& buffer)
{
    resize(buffer, 0);
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::alloc(DeviceBuffer<T>& buffer)
{
    MUDA_ASSERT(!buffer.m_data, "The buffer is already allocated");
    BufferLaunch().resize(buffer, n);
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::free(DeviceBuffer<T>& buffer)
{
    auto& m_data     = buffer.m_data;
    auto& m_size     = buffer.m_size;
    auto& m_capacity = buffer.m_capacity;

    MUDA_ASSERT(buffer.m_data, "The buffer is not allocated");

    Memory(m_stream).free(m_data);
    m_data     = nullptr;
    m_size     = 0;
    m_capacity = 0;
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::shrink_to_fit(DeviceBuffer<T>& buffer)
{
    auto  mem        = Memory(m_stream);
    auto& m_data     = buffer.m_data;
    auto& m_size     = buffer.m_size;
    auto& m_capacity = buffer.m_capacity;
    if(m_size < m_capacity)
    {
        T* ptr = nullptr;
        if(m_size > 0)
        {
            mem.alloc(&ptr, m_size * sizeof(T));
            mem.transfer(ptr, m_data, m_size * sizeof(T));
        }
        if(m_data)
            mem.free(m_data);
        m_data     = ptr;
        m_capacity = m_size;
    }
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::copy(DeviceBufferView<T>& dst, const DeviceBufferView<T>& src)
{
    MUDA_ASSERT(dst.size() == src.size(), "BufferView should have the same size");
    if constexpr(std::is_trivially_copyable_v<T>)
        Memory(m_stream).transfer(dst.data(), src.data(), src.size() * sizeof(T));
    else
        details::kernel_assign(
            m_grid_dim, m_block_dim, m_stream, dst.data(), src.data(), src.size());
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::copy(T* dst, const DeviceBufferView<T>& src)
{
    Memory(m_stream).download(dst, src.data(), src.size() * sizeof(T));
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::copy(DeviceBufferView<T>& dst, const T* src)
{
    Memory(m_stream).upload(dst.data(), src, dst.size() * sizeof(T));
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::copy(DeviceBufferVar<T>& dst, const DeviceBufferVar<T>& src)
{
    if constexpr(std::is_trivially_copyable_v<T>)
        Memory(m_stream).transfer(dst.data(), src.data(), sizeof(T));
    else
        details::kernel_assign(1, 1, m_stream, dst.data(), src.data(), 1);
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::copy(T* dst, const DeviceBufferVar<T>& src)
{
    Memory(m_stream).download(dst, src.data(), sizeof(T));
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::copy(DeviceBufferVar<T>& dst, const T* src)
{
    Memory(m_stream).upload(dst.data(), src, sizeof(T));
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::copy(DeviceBufferVar<T>& dst, const DeviceBufferView<T>& src)
{
    MUDA_ASSERT(src.size() == 1, "To copy to DeviceBufferVar, BufferView should have the size of 1");
    Memory(m_stream).transfer(dst.data(), src.data(), sizeof(T));
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::copy(DeviceBufferView<T>& dst, const DeviceBufferVar<T>& src)
{
    MUDA_ASSERT(dst.size() == 1, "To copy from DeviceBufferVar, BufferView should have the size of 1");
    Memory(m_stream).transfer(dst.data(), src.data(), sizeof(T));
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::fill(DeviceBufferView<T>& buffer, const T& val)
{
    ParallelFor(m_grid_dim, m_block_dim, 0, m_stream)
        .apply(buffer.size(),
               [d = buffer.data(), val] __device__(int i) mutable { d[i] = val; });
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::fill(DeviceBufferVar<T>& buffer, const T& val)
{
    if constexpr(std::is_trivially_copyable_v<T>)
    {
        Memory(m_stream).upload(buffer.data(), &val, sizeof(T));
    }
    else
    {
        details::kernel_fill(1, 1, m_stream, buffer.data(), val, 1);
    }
    return *this;
}


template <typename T, typename FConstruct>
BufferLaunch& BufferLaunch::resize(DeviceBuffer<T>& buffer, size_t new_size, FConstruct&& fct)
{
    auto mem = Memory(m_stream);

    auto&   m_data     = buffer.m_data;
    size_t& m_size     = buffer.m_size;
    size_t& m_capacity = buffer.m_capacity;

    if(new_size == m_size)
        return *this;

    size_t old_size = buffer.m_size;

    if(new_size < m_size)
    {
        // destruct the old memory
        if constexpr(!std::is_trivially_destructible_v<T>)
        {
            details::kernel_destruct(
                m_grid_dim, m_block_dim, m_stream, m_data + new_size, m_size - new_size);
        }
        m_size = new_size;
        return *this;
    }

    if(new_size <= m_capacity)
    {
        // zero out the new memory
        mem.set(m_data + old_size, (new_size - old_size) * sizeof(T), 0);
        m_size = new_size;
    }
    else
    {
        T* ptr;
        mem.alloc(&ptr, new_size * sizeof(T));
        if(m_data)
            mem.transfer(ptr, m_data, old_size * sizeof(T));

        // construct the new memory
        fct(ptr);

        if(m_data)
            mem.free(m_data);

        m_data     = ptr;
        m_size     = new_size;
        m_capacity = new_size;
    }
    return *this;
}

}  // namespace muda