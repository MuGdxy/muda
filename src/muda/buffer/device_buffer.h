#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <optional>
#include "../container/vector.h"
#include "../container/var.h"

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
  public:
    using value_type = T;

    device_buffer(cudaStream_t s, size_t n)
        : stream_(s)
        , init_(true)
    {
        memory(stream_).alloc(&data_, n * sizeof(value_type));
        size_     = n;
        capacity_ = n;
    }

    device_buffer()
        : stream_(nullptr)
        , data_(nullptr)
        , size_(0)
        , capacity_(0)
        , init_(false){};

    explicit device_buffer(cudaStream_t s)
        : stream_(s)
        , data_(nullptr)
        , size_(0)
        , capacity_(0)
        , init_(true){};

    device_buffer(const device_buffer& other) = delete;

    device_buffer(device_buffer&& other) noexcept
        : stream_(other.stream_)
        , data_(other.data_)
        , size_(other.size_)
        , capacity_(other.capacity_)
        , init_(other.init_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.init_ = false;
    }

    device_buffer& operator=(const device_buffer& other) = delete;

    void stream(cudaStream_t s)
    {
        init_   = true;
        stream_ = s;
    }
    cudaStream_t stream() { return stream_; }

    //void resize(size_t new_size, buf_op mem_op = buf_op::keep_set, char setbyte = 0)
    //{
    //    init_         = true;
    //    auto old_size = size_;
    //    auto mem      = memory(stream_);
    //    if(new_size == 0)
    //        throw std::logic_error("new_size = 0 is not allowed");

    //    if(old_size < new_size)  // expand
    //    {
    //        T* ptr;
    //        mem.alloc(&ptr, new_size * sizeof(value_type));
    //        switch(mem_op)
    //        {
    //            case muda::buf_op::keep:
    //                mem.copy(ptr, data_, old_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
    //                break;
    //            case muda::buf_op::set:
    //                mem.set(ptr, (int)setbyte, new_size * sizeof(value_type));
    //                break;
    //            case muda::buf_op::keep_set:
    //                if(data_)
    //                    mem.copy(ptr, data_, old_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
    //                mem.set(ptr + old_size,
    //                        (int)setbyte,
    //                        (new_size - old_size) * sizeof(value_type));
    //                break;
    //            default:
    //                break;
    //        }
    //        if(data_)
    //            mem.free(data_);
    //        data_ = ptr;
    //        size_ = new_size;
    //    }
    //    else if(old_size > new_size)  // shrink
    //    {
    //        T* ptr;
    //        mem.alloc(&ptr, new_size * sizeof(value_type));
    //        switch(mem_op)
    //        {
    //            case muda::buf_op::keep:
    //                mem.copy(ptr, data_, new_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
    //                break;
    //            case muda::buf_op::set:
    //                mem.set(ptr, (int)setbyte, sizeof(value_type) * new_size);
    //                break;
    //            default:
    //                break;
    //        }
    //        if(data_)
    //            mem.free(data_);
    //        data_ = ptr;
    //        size_ = new_size;
    //    }
    //    else  // keep
    //    {
    //        if(mem_op == buf_op::set)
    //            mem.set(data_, (int)setbyte, sizeof(value_type) * new_size);
    //    }
    //}

    empty resize(size_t new_size, buf_op mem_op = buf_op::keep_set, char setbyte = 0)
    {
        auto   mem      = memory(stream_);
        size_t old_size = size_;

        if(new_size <= size_)
        {
            switch(mem_op)
            {
                case muda::buf_op::set:
                    mem.set(data_, new_size * sizeof(value_type), (int)setbyte);
                    break;
                default:
                    break;
            }
            size_ = new_size;
            return empty(stream_);
        }

        if(new_size <= capacity_)
        {
            switch(mem_op)
            {
                case muda::buf_op::set:
                    mem.set(data_, new_size * sizeof(value_type), (int)setbyte);
                    break;
                case muda::buf_op::keep_set:
                    mem.set(data_ + old_size,
                            (new_size - old_size) * sizeof(value_type),
                            (int)setbyte);
                    break;
                default:
                    break;
            }
            size_ = new_size;
        }
        else
        {
            T* ptr;
            mem.alloc(&ptr, new_size * sizeof(value_type));
            switch(mem_op)
            {
                case muda::buf_op::keep:
                    mem.copy(ptr, data_, old_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
                    break;
                case muda::buf_op::set:
                    mem.set(ptr, new_size * sizeof(value_type), (int)setbyte);
                    break;
                case muda::buf_op::keep_set:
                    if(data_)
                        mem.copy(ptr, data_, old_size * sizeof(value_type), cudaMemcpyDeviceToDevice);
                    mem.set(ptr + old_size,
                            (new_size - old_size) * sizeof(value_type),
                            (int)setbyte);
                    break;
                default:
                    break;
            }
            if(data_)
                mem.free(data_);
            data_     = ptr;
            size_     = new_size;
            capacity_ = new_size;
        }

        return empty(stream_);
    }

    empty shrink_to_fit()
    {
        auto mem = memory(stream_);

        if(size_ < capacity_)
        {
            T* ptr;
            mem.alloc(&ptr, size_ * sizeof(value_type));
            mem.copy(ptr, data_, size_ * sizeof(value_type), cudaMemcpyDeviceToDevice);
            if(data_)
                mem.free(data_);
            data_     = ptr;
            capacity_ = size_;
        }

        return empty(stream_);
    }

    empty set(char setbyte = 0, size_t count = size_t(-1))
    {
        init_ = true;
        if(count == size_t(-1))
            count = size_;
        if(count > size_)
            throw std::out_of_range("device_buffer::set out of range");
        memory(stream_).set(data_, count * sizeof(T), setbyte);
        return empty(stream_);
    }

    // copy to/from
    empty copy_to(value_type& var) const
    {
        if(size_ != 1)
            throw std::logic_error("buffer size larger than 1, cannot copy to host_var");
        init_ = true;
        memory(stream_).copy(std::addressof(var), data_, size_ * sizeof(value_type), cudaMemcpyDeviceToHost);
        return empty(stream_);
    }

    empty copy_to(host_vector<value_type>& vec) const
    {
        init_ = true;
        vec.resize(size_);
        memory(stream_).copy(muda::data(vec), data_, size_ * sizeof(value_type), cudaMemcpyDeviceToHost);
        return empty(stream_);
    }

    empty copy_to(device_var<value_type>& var) const
    {
        if(size_ != 1)
            throw std::logic_error("buffer size larger than 1, cannot copy to device_var");
        init_ = true;
        memory(stream_).copy(muda::data(var), data_, size_ * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(stream_);
    }

    empty copy_to(device_vector<value_type>& vec) const
    {
        init_ = true;
        vec.resize(size_);
        memory(stream_).copy(muda::data(vec), data_, size_ * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(stream_);
    }


    empty copy_from(const host_var<value_type>& var)
    {
        init_ = true;
        resize(1);
        memory(stream_).copy(data_, muda::data(var), size_ * sizeof(value_type), cudaMemcpyHostToDevice);
        return empty(stream_);
    }

    empty copy_from(const value_type& var)
    {
        init_ = true;
        resize(1);
        memory(stream_).copy(data_, muda::data(var), size_ * sizeof(value_type), cudaMemcpyHostToDevice);
        return empty(stream_);
    }

    empty copy_from(const host_vector<value_type>& vec)
    {
        init_ = true;
        resize(vec.size());
        memory(stream_).copy(data_, muda::data(vec), size_ * sizeof(value_type), cudaMemcpyHostToDevice);
        return empty(stream_);
    }

    empty copy_from(const device_var<value_type>& var)
    {
        init_ = true;
        resize(1);
        memory(stream_).copy(data_, muda::data(var), size_ * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(stream_);
    }

    empty copy_from(const device_vector<value_type>& vec)
    {
        init_ = true;
        resize(vec.size());
        memory().copy(data_, muda::data(vec), size_ * sizeof(value_type), cudaMemcpyDeviceToDevice);
        return empty(stream_);
    }

    ~device_buffer()
    {
        if(data_)
            memory(stream_).free(data_);
    }

    size_t   size() const { return size_; }
    T*       data() { return data_; }
    const T* data() const { return data_; }
    bool     already_init() const { return init_; }

  private:
    mutable bool init_;
    cudaStream_t stream_;
    size_t       size_;
    size_t       capacity_;
    T*           data_;
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
inline __host__ auto make_idxer(device_buffer<T>& buf) noexcept
{
    return idxer1D<T>(buf.data(), buf.size());
}

template <typename T>
inline __host__ auto make_idxer2D(device_buffer<T>& buf, uint32_t dimx, uint32_t dimy) noexcept
{
    assert(dimx * dimy <= buf.size());
    return idxer2D<T>(buf.data(), dimx, dimy);
}

template <typename T>
inline __host__ auto make_idxer3D(device_buffer<T>& buf, uint32_t dimx, uint32_t dimy, uint32_t dimz) noexcept
{
    assert(dimx * dimy * dimz <= buf.size());
    return idxer3D<T>(buf.data(), dimx, dimy, dimz);
}

template <typename T>
inline __host__ auto make_viewer(device_buffer<T>& buf) noexcept
{
    return make_idxer(buf);
}
}  // namespace muda