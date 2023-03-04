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

    device_buffer(device_buffer&& other) MUDA_NOEXCEPT
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

    empty resize(size_t new_size, buf_op mem_op, char setbyte = 0);

    empty resize(size_t new_size);

    empty resize(size_t new_size, const value_type& value, int blockDim = LIGHT_WORKLOAD_BLOCK_SIZE);

    empty shrink_to_fit();

    empty set(char setbyte = 0, size_t count = size_t(-1));

    empty fill(const T& v, size_t count = size_t(-1), int blockDim = LIGHT_WORKLOAD_BLOCK_SIZE);

    // copy to/from
    empty copy_to(value_type& var) const;

    empty copy_to(host_vector<value_type>& vec) const;

    empty copy_to(device_var<value_type>& var) const;

    empty copy_to(device_vector<value_type>& vec) const;

    empty copy_to(device_buffer<value_type>& vec) const;

    empty copy_from(const host_var<value_type>& var);

    empty copy_from(const value_type& var);

    empty copy_from(const host_vector<value_type>& vec);

    empty copy_from(const device_var<value_type>& var);

    empty copy_from(const device_vector<value_type>& vec);

    empty copy_from(const device_buffer<value_type>& vec);

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


class buffer_launch : public launch_base<buffer_launch>
{
    int m_blockDim;

  public:
    buffer_launch(int blockDim = LIGHT_WORKLOAD_BLOCK_SIZE, cudaStream_t stream = nullptr)
        : launch_base(stream)
        , m_blockDim(blockDim)
    {
    }

    template <typename T>
    buffer_launch& resize(device_buffer<T>& buf, size_t size)
    {
        details::set_stream_check(buf, m_stream);
        buf.resize(size);
        return *this;
    }

    template <typename T>
    buffer_launch& resize(device_buffer<T>& buf, size_t size, const T& value)
    {
        details::set_stream_check(buf, m_stream);
        buf.resize(size, value, m_blockDim);
        return *this;
    }

    template <typename T>
    buffer_launch& resize(device_buffer<T>& buf, size_t size, buf_op mem_op, char setbyte = 0)
    {
        details::set_stream_check(buf, m_stream);
        buf.resize(size, mem_op, setbyte);
        return *this;
    }

    template <typename T>
    buffer_launch& shrink_to_fit(device_buffer<T>& buf)
    {
        details::set_stream_check(buf, m_stream);
        buf.shrink_to_fit();
        return *this;
    }

    template <typename T>
    buffer_launch& set(device_buffer<T>& buf, char setbyte = 0, size_t count = size_t(-1))
    {
        details::set_stream_check(buf, m_stream);
        buf.set(setbyte, count);
        return *this;
    }

    template <typename T>
    buffer_launch& copy_to(device_buffer<T>& buf, T& val)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_to(val);
        return *this;
    }

    template <typename T>
    buffer_launch& copy_to(device_buffer<T>& buf, host_vector<T>& vec)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_to(vec);
        return *this;
    }

    template <typename T>
    buffer_launch& copy_to(device_buffer<T>& buf, device_var<T>& var)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_to(var);
        return *this;
    }

    template <typename T>
    buffer_launch& copy_to(device_buffer<T>& buf, device_vector<T>& vec)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_to(vec);
        return *this;
    }

    template <typename T>
    buffer_launch& copy_to(device_buffer<T>& buf, device_buffer<T>& dst)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_to(dst);
        return *this;
    }


    template <typename T>
    buffer_launch& copy_from(device_buffer<T>& buf, const T& val)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_from(val);
        return *this;
    }

    template <typename T>
    buffer_launch& copy_from(device_buffer<T>& buf, const host_vector<T>& vec)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_from(vec);
        return *this;
    }

    template <typename T>
    buffer_launch& copy_from(device_buffer<T>& buf, device_var<T>& var)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_from(var);
        return *this;
    }

    template <typename T>
    buffer_launch& copy_from(device_buffer<T>& buf, const device_vector<T>& vec)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_from(vec);
        return *this;
    }

    template <typename T>
    buffer_launch& copy_from(device_buffer<T>& buf, const device_buffer<T>& vec)
    {
        details::set_stream_check(buf, m_stream);
        buf.copy_from(vec);
        return *this;
    }
};
}  // namespace muda

namespace muda
{
template <typename T>
MUDA_INLINE MUDA_HOST auto data(device_buffer<T>& buf) MUDA_NOEXCEPT
{
    return buf.data();
}

template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense(device_buffer<T>& buf) MUDA_NOEXCEPT
{
    return dense1D<T>(buf.data(), buf.size());
}

template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(device_buffer<T>& buf, uint32_t dimx, uint32_t dimy) MUDA_NOEXCEPT
{
    assert(dimx * dimy <= buf.size());
    return dense2D<T>(buf.data(), dimx, dimy);
}

template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(device_buffer<T>& buf, uint32_t dimx, uint32_t dimy, uint32_t dimz) MUDA_NOEXCEPT
{
    assert(dimx * dimy * dimz <= buf.size());
    return dense3D<T>(buf.data(), dimx, dimy, dimz);
}

template <typename T>
MUDA_INLINE MUDA_HOST auto make_viewer(device_buffer<T>& buf) MUDA_NOEXCEPT
{
    return make_dense(buf);
}
}  // namespace muda

#include "device_buffer.inl"