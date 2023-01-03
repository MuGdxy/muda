#pragma once
#include <thrust/device_allocator.h>
#include <thrust/universal_allocator.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <vector>

namespace muda
{
namespace details
{
    template <typename T, typename Allocator>
    class var_base
    {
      public:
        //using T = int;
        //using Allocator = thrust::device_allocator<int>;
        using pointer       = typename Allocator::pointer;
        using const_pointer = typename Allocator::const_pointer;

        __host__ var_base() noexcept
            : ptr(Allocator().allocate(1))
        {
        }

        __host__ var_base(const T& value) noexcept
            : ptr(Allocator().allocate(1))
        {
            this->operator=(value);
        }

        __host__ ~var_base() noexcept { Allocator().deallocate(ptr, 1); }

        pointer       data() { return ptr; }
        const_pointer data() const { return ptr; }

        // copy value from host to device
        __host__ var_base& operator=(const T& rhs)
        {
            thrust::fill_n(data(), 1, rhs);
            return *this;
        }

        // copy value from device to host
        __host__ operator T()
        {
            T t;
            thrust::copy_n(data(), 1, &t);
            return t;
        }

      private:
        pointer ptr;
    };
}  // namespace details

template <typename T, typename Allocator = thrust::device_allocator<T>>
using device_var = details::var_base<T, Allocator>;

template <typename T, typename Allocator = thrust::universal_allocator<T>>
using universal_var = details::var_base<T, Allocator>;

template <typename T, typename Allocator = std::allocator<T>>
using host_var = details::var_base<T, Allocator>;
}  // namespace muda

namespace muda
{
//template <typename T>
//inline const T* data(const T& v) noexcept
//{
//    return std::addressof(v);
//}
//
//template <typename T>
//inline T* data(T& v) noexcept
//{
//    return std::addressof(v);
//}

template <typename T, typename Allocator>
inline const T* data(const details::var_base<T, Allocator>& v) noexcept
{
    return thrust::raw_pointer_cast(v.data());
}

template <typename T, typename Allocator>
inline T* data(details::var_base<T, Allocator>& v) noexcept
{
    return thrust::raw_pointer_cast(v.data());
}
}  // namespace muda

#include <muda/viewer/idxer.h>
namespace muda
{
template <typename T, typename Allocator>
inline __host__ auto make_idxer(details::var_base<T, Allocator>& v) noexcept
{
    return idxer<T>(data(v));
}

template <typename T, typename Allocator>
inline __host__ auto make_viewer(details::var_base<T, Allocator>& v) noexcept
{
    return make_idxer(v);
}

//print convert
template <typename T>
inline __host__ __device__ const T& printConvert(const idxer<T>& idx)
{
    return idx.operator const T&();
}
}  // namespace muda