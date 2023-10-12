#include <muda/buffer/device_buffer.h>

namespace muda
{
MUDA_INLINE BufferLaunch::BufferLaunch(cudaStream_t s)
    : LaunchBase(s)
{
}

MUDA_INLINE BufferLaunch::BufferLaunch(int block_dim, cudaStream_t s)
    : LaunchBase(s)
    , m_block_dim(block_dim)
{
}

MUDA_INLINE BufferLaunch::BufferLaunch(int grid_dim, int block_dim, cudaStream_t s)
    : LaunchBase(s)
    , m_grid_dim(grid_dim)
    , m_block_dim(block_dim)
{
}

template <typename T, typename F>
BufferLaunch& muda::BufferLaunch::resize(DeviceBuffer<T>& buffer, int size, F&& f)
{
    auto mem = Memory(m_stream);

    size_t& m_data     = buffer.m_data;
    size_t& m_size     = buffer.m_size;
    size_t& m_capacity = buffer.m_capacity;

    size_t old_size = buffer.m_size;

    if(new_size <= m_size)
    {
        m_size = new_size;
        return *this;
    }

    if(new_size <= m_capacity)
    {
        // zero out the new memory
        mem.set(m_data + old_size, (new_size - old_size) * sizeof(value_type), 0);
        m_size = new_size;
    }
    else
    {
        T* ptr;
        mem.alloc(&ptr, new_size * sizeof(value_type));
        if(m_data)
            mem.transfer(ptr, m_data, old_size * sizeof(value_type));

        // fill the new memory
        f();

        if(m_data)
            mem.free(m_data);
        m_data     = ptr;
        m_size     = new_size;
        m_capacity = new_size;
    }
    return *this;
}

template <typename T>
BufferLaunch& BufferLaunch::resize(DeviceBuffer<T>& buffer, int new_size)
{
    return resize(buffer,
                  new_size,
                  []
                  {
                      Memory(m_stream).set(ptr + old_size,
                                           (new_size - old_size) * sizeof(value_type),
                                           0);
                  });
}

template <typename T>
BufferLaunch& BufferLaunch::resize(DeviceBuffer<T>& buffer, int new_size, const T& value)
{
    return resize(buffer,
                  new_size,
                  []
                  {
                      ParallelFor(m_grid_dim, m_block_dim, 0, m_stream)
                          .apply(new_size - old_size,
                                 [=, d = this->viewer()] __device__(int i) mutable
                                 { d(old_size + i) = value; });
                  });
}
}  // namespace muda