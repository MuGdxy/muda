#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <optional>

#include <muda/container/vector.h>
#include <muda/container/var.h>
#include <muda/launch/launch_base.h>
#include <muda/launch/memory.h>
#include <muda/launch/parallel_for.h>

namespace muda
{
enum class buf_op : unsigned
{
    ignore,
    keep     = 1 << 0,
    set      = 1 << 1,
    keep_set = keep | set
};

template <typename T = std::byte>
class device_buffer
{
  private:
    mutable bool m_init;
    cudaStream_t m_stream;
    size_t       m_size;
    size_t       m_capacity;
    T*           m_data;
    
  public:
    using value_type = T;

    device_buffer(cudaStream_t s, size_t n)
        : m_stream(s)
        , m_init(true)
    {
        memory(m_stream).alloc(&m_data, n * sizeof(value_type));
        m_size     = n;
        m_capacity = n;
    }

    device_buffer()
        : m_stream(nullptr)
        , m_data(nullptr)
        , m_size(0)
        , m_capacity(0)
        , m_init(false){};

    explicit device_buffer(cudaStream_t s)
        : m_stream(s)
        , m_data(nullptr)
        , m_size(0)
        , m_capacity(0)
        , m_init(true){};

    device_buffer(const device_buffer& other) = delete;

    device_buffer(device_buffer&& other) noexcept
        : m_stream(other.m_stream)
        , m_data(other.m_data)
        , m_size(other.m_size)
        , m_capacity(other.m_capacity)
        , m_init(other.m_init)
    {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_init = false;
    }

    device_buffer& operator=(const device_buffer& other) = delete;

    void stream(cudaStream_t s)
    {
        m_init   = true;
        m_stream = s;
    }
    cudaStream_t stream() { return m_stream; }

    empty resize(size_t new_size, buf_op mem_op, char setbyte = 0)
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
                    mem.set(ptr + old_size,
                            (new_size - old_size) * sizeof(value_type),
                            (int)setbyte);
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

    empty resize(size_t new_size)
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

    empty resize(size_t new_size, const T& value)
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

        parallel_for(LIGHT_WORKLOAD_BLOCK_SIZE, 0, m_stream)
            .apply(new_size,
                   [=, d = make_viewer(*this)] __device__(int i) mutable
                   { d(i) = value; });

        return empty(m_stream);
    }

    empty shrink_to_fit()
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

    empty set(char setbyte = 0, size_t count = size_t(-1))
    {
        m_init = true;
        if(count == size_t(-1))
            count = m_size;
        if(count > m_size)
            throw std::out_of_range("device_buffer::set out of range");
        memory(m_stream).set(m_data, count * sizeof(T), setbyte);
        return empty(m_stream);
    }

    // copy to/from
    empty copy_to(value_type& var) const
    {
        if(m_size != 1)
            throw std::logic_error("buffer size larger than 1, cannot copy to host_var");
        m_init = true;
        memory(m_stream).copy(std::addressof(var), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToHost);
        return empty(m_stream);
    }

    empty copy_to(host_vector<value_type>& vec) const
    {
        m_init = true;
        vec.resize(m_size);
        memory(m_stream).copy(muda::data(vec), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToHost);
        return empty(m_stream);
    }

    empty copy_to(device_var<value_type>& var) const
    {
        if(m_size != 1)
            throw std::logic_error("buffer size larger than 1, cannot copy to device_var");
        m_init = true;
        memory(m_stream).copy(muda::data(var), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(m_stream);
    }

    empty copy_to(device_vector<value_type>& vec) const
    {
        m_init = true;
        vec.resize(m_size);
        memory(m_stream).copy(muda::data(vec), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(m_stream);
    }

    empty copy_to(device_buffer<value_type>& vec) const
    {
        m_init = true;
        vec.resize(m_size);
        memory(m_stream).copy(vec.data(), m_data, m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(m_stream);
    }

    empty copy_from(const host_var<value_type>& var)
    {
        m_init = true;
        resize(1);
        memory(m_stream).copy(m_data, muda::data(var), m_size * sizeof(value_type), cudaMemcpyHostToDevice);
        return empty(m_stream);
    }

    empty copy_from(const value_type& var)
    {
        m_init = true;
        resize(1);
        memory(m_stream).copy(m_data, std::addressof(var), m_size * sizeof(value_type), cudaMemcpyHostToDevice);
        return empty(m_stream);
    }

    empty copy_from(const host_vector<value_type>& vec)
    {
        m_init = true;
        resize(vec.size());
        memory(m_stream).copy(m_data, muda::data(vec), m_size * sizeof(value_type), cudaMemcpyHostToDevice);
        return empty(m_stream);
    }

    empty copy_from(const device_var<value_type>& var)
    {
        m_init = true;
        resize(1);
        memory(m_stream).copy(m_data, muda::data(var), m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(m_stream);
    }

    empty copy_from(const device_vector<value_type>& vec)
    {
        m_init = true;
        resize(vec.size());
        memory().copy(m_data, muda::data(vec), m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(m_stream);
    }

    empty copy_from(const device_buffer<value_type>& vec)
    {
        m_init = true;
        resize(vec.size());
        memory().copy(m_data, vec.data(), m_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(m_stream);
    }

    ~device_buffer()
    {
        if(m_data)
            memory(m_stream).free(m_data);
    }

    size_t   size() const { return m_size; }
    T*       data() { return m_data; }
    const T* data() const { return m_data; }
    bool     already_init() const { return m_init; }
};

namespace details
{
    template <typename T = std::byte>
    void set_stream_check(device_buffer<T>& buf, cudaStream_t s)
    {
        if(buf.already_init() && s != buf.stream())
            throw std::logic_error("buffer is already initialized, please manually set the buffer's stream to s");
        buf.stream(s);  // buffer isn't initialized yet, allows any setting.
    }
}  // namespace details
}  // namespace muda

namespace muda
{
template <typename T>
inline __host__ auto data(device_buffer<T>& buf) noexcept
{
    return buf.data();
}

template <typename T>
inline __host__ auto make_dense(device_buffer<T>& buf) noexcept
{
    return dense1D<T>(buf.data(), buf.size());
}

template <typename T>
inline __host__ auto make_dense2D(device_buffer<T>& buf, uint32_t dimx, uint32_t dimy) noexcept
{
    assert(dimx * dimy <= buf.size());
    return dense2D<T>(buf.data(), dimx, dimy);
}

template <typename T>
inline __host__ auto make_dense3D(device_buffer<T>& buf, uint32_t dimx, uint32_t dimy, uint32_t dimz) noexcept
{
    assert(dimx * dimy * dimz <= buf.size());
    return dense3D<T>(buf.data(), dimx, dimy, dimz);
}

template <typename T>
inline __host__ auto make_viewer(device_buffer<T>& buf) noexcept
{
    return make_dense(buf);
}
}  // namespace muda