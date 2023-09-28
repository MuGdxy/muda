//#pragma once
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <optional>
//
//#include <muda/container/vector.h>
//#include <muda/container/var.h>
//#include <muda/launch/launch_base.h>
//#include <muda/launch/memory.h>
//#include <muda/launch/parallel_for.h>
//
//namespace muda
//{
//enum class BufferOperation : unsigned
//{
//    ignore,
//    keep     = 1 << 0,
//    set      = 1 << 1,
//    keep_set = keep | set
//};
//
//template <typename T = std::byte>
//class DeviceBuffer
//{
//  private:
//    mutable bool m_init;
//    cudaStream_t m_stream;
//    size_t       m_size;
//    size_t       m_capacity;
//    T*           m_data;
//
//  public:
//    using value_type = T;
//
//    DeviceBuffer(cudaStream_t s, size_t n)
//        : m_stream(s)
//        , m_init(true)
//    {
//        Memory(m_stream).alloc(&m_data, n * sizeof(value_type));
//        m_size     = n;
//        m_capacity = n;
//    }
//
//    DeviceBuffer()
//        : m_stream(nullptr)
//        , m_data(nullptr)
//        , m_size(0)
//        , m_capacity(0)
//        , m_init(false){};
//
//    explicit DeviceBuffer(cudaStream_t s)
//        : m_stream(s)
//        , m_data(nullptr)
//        , m_size(0)
//        , m_capacity(0)
//        , m_init(true){};
//
//    DeviceBuffer(const DeviceBuffer& other) = delete;
//
//    DeviceBuffer(DeviceBuffer&& other) MUDA_NOEXCEPT
//        : m_stream(other.m_stream),
//          m_data(other.m_data),
//          m_size(other.m_size),
//          m_capacity(other.m_capacity),
//          m_init(other.m_init)
//    {
//        other.m_data = nullptr;
//        other.m_size = 0;
//        other.m_init = false;
//    }
//
//    DeviceBuffer& operator=(const DeviceBuffer& other) = delete;
//
//    void stream(cudaStream_t s)
//    {
//        m_init         = true;
//        m_stream = s;
//    }
//
//    cudaStream_t stream() { return m_stream; }
//
//    Empty resize(size_t new_size, BufferOperation mem_op, char setbyte = 0);
//
//    Empty resize(size_t new_size);
//
//    Empty resize(size_t new_size, const value_type& value, int blockDim = LIGHT_WORKLOAD_BLOCK_SIZE);
//
//    Empty shrink_to_fit();
//
//    Empty set(char setbyte = 0, size_t count = size_t(-1));
//
//    Empty fill(const T& v, size_t count = size_t(-1), int blockDim = LIGHT_WORKLOAD_BLOCK_SIZE);
//
//    // copy to/from
//    Empty copy_to(value_type& var) const;
//
//    Empty copy_to(HostVector<value_type>& vec) const;
//
//    Empty copy_to(DeviceVar<value_type>& var) const;
//
//    Empty copy_to(DeviceVector<value_type>& vec) const;
//
//    Empty copy_to(DeviceBuffer<value_type>& vec) const;
//
//    Empty copy_from(const value_type& var);
//
//    Empty copy_from(const HostVector<value_type>& vec);
//
//    Empty copy_from(const DeviceVar<value_type>& var);
//
//    Empty copy_from(const DeviceVector<value_type>& vec);
//
//    Empty copy_from(const DeviceBuffer<value_type>& vec);
//
//    ~DeviceBuffer()
//    {
//        if(m_data)
//            Memory(this->stream()).free(m_data);
//    }
//
//    size_t   size() const { return m_size; }
//    T*       data() { return m_data; }
//    const T* data() const { return m_data; }
//    bool     already_init() const { return m_init; }
//};
//
//namespace details
//{
//    template <typename T = std::byte>
//    void set_stream_check(DeviceBuffer<T>& buf, cudaStream_t s)
//    {
//        if(buf.already_init() && s != buf.stream())
//            MUDA_ERROR_WITH_LOCATION("buffer is already initialized, please manually set the buffer's stream to s");
//        buf.stream(s);  // buffer isn't initialized yet, allows any setting.
//    }
//}  // namespace details
//
//
//class BufferOperator : public LaunchBase<BufferOperator>
//{
//    int m_block_dim;
//
//  public:
//    BufferOperator(int blockDim = LIGHT_WORKLOAD_BLOCK_SIZE, cudaStream_t stream = nullptr)
//        : LaunchBase(stream)
//        , m_block_dim(blockDim)
//    {
//    }
//
//    template <typename T>
//    BufferOperator& resize(DeviceBuffer<T>& buf, size_t size)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.resize(size);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& resize(DeviceBuffer<T>& buf, size_t size, const T& value)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.resize(size, value, m_block_dim);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& resize(DeviceBuffer<T>& buf, size_t size, BufferOperation mem_op, char setbyte = 0)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.resize(size, mem_op, setbyte);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& shrink_to_fit(DeviceBuffer<T>& buf)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.shrink_to_fit();
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& set(DeviceBuffer<T>& buf, char setbyte = 0, size_t count = size_t(-1))
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.set(setbyte, count);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& copy_to(DeviceBuffer<T>& buf, T& val)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_to(val);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& copy_to(DeviceBuffer<T>& buf, HostVector<T>& vec)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_to(vec);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& copy_to(DeviceBuffer<T>& buf, DeviceVar<T>& var)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_to(var);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& copy_to(DeviceBuffer<T>& buf, DeviceVector<T>& vec)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_to(vec);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& copy_to(DeviceBuffer<T>& buf, DeviceBuffer<T>& dst)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_to(dst);
//        return *this;
//    }
//
//
//    template <typename T>
//    BufferOperator& copy_from(DeviceBuffer<T>& buf, const T& val)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_from(val);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& copy_from(DeviceBuffer<T>& buf, const HostVector<T>& vec)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_from(vec);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& copy_from(DeviceBuffer<T>& buf, DeviceVar<T>& var)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_from(var);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& copy_from(DeviceBuffer<T>& buf, const DeviceVector<T>& vec)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_from(vec);
//        return *this;
//    }
//
//    template <typename T>
//    BufferOperator& copy_from(DeviceBuffer<T>& buf, const DeviceBuffer<T>& vec)
//    {
//        details::set_stream_check(buf, this->stream());
//        buf.copy_from(vec);
//        return *this;
//    }
//};
//}  // namespace muda
//
//namespace muda
//{
//template <typename T>
//MUDA_INLINE MUDA_HOST auto data(DeviceBuffer<T>& buf) MUDA_NOEXCEPT
//{
//    return buf.data();
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense(DeviceBuffer<T>& buf) MUDA_NOEXCEPT
//{
//    return Dense1D<T>(buf.data(), buf.size());
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense2D(DeviceBuffer<T>& buf, uint32_t dimx, uint32_t dimy) MUDA_NOEXCEPT
//{
//    assert(dimx * dimy <= buf.size());
//    return Dense2D<T>(buf.data(), dimx, dimy);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(DeviceBuffer<T>& buf, uint32_t dimx, uint32_t dimy, uint32_t dimz) MUDA_NOEXCEPT
//{
//    assert(dimx * dimy * dimz <= buf.size());
//    return Dense3D<T>(buf.data(), dimx, dimy, dimz);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_viewer(DeviceBuffer<T>& buf) MUDA_NOEXCEPT
//{
//    return make_dense(buf);
//}
//}  // namespace muda
//
//#include "device_buffer.inl"