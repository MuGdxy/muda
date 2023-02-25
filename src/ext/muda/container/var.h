#pragma once
#include <vector>
#include <thrust/device_allocator.h>
#include <thrust/universal_allocator.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <muda/muda_def.h>

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

        MUDA_HOST var_base() MUDA_NOEXCEPT
            : m_data(Allocator().allocate(1))
        {
        }

        MUDA_HOST var_base(const T& value) MUDA_NOEXCEPT
            : m_data(Allocator().allocate(1))
        {
            this->operator=(value);
        }

        MUDA_HOST ~var_base() MUDA_NOEXCEPT { Allocator().deallocate(m_data, 1); }

        pointer       data() { return m_data; }
        const_pointer data() const { return m_data; }

        // copy value from host to device
        MUDA_HOST var_base& operator=(const T& rhs)
        {
            thrust::fill_n(data(), 1, rhs);
            return *this;
        }

        // copy value from device to host
        MUDA_HOST operator T()
        {
            T t;
            thrust::copy_n(data(), 1, &t);
            return t;
        }

      private:
        pointer m_data;
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
template <typename T, typename Allocator>
MUDA_INLINE const T* data(const details::var_base<T, Allocator>& v) MUDA_NOEXCEPT
{
    return thrust::raw_pointer_cast(v.data());
}

template <typename T, typename Allocator>
MUDA_INLINE T* data(details::var_base<T, Allocator>& v) MUDA_NOEXCEPT
{
    return thrust::raw_pointer_cast(v.data());
}
}  // namespace muda

#include <muda/viewer/dense.h>
namespace muda
{
template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense(details::var_base<T, Allocator>& v) MUDA_NOEXCEPT
{
    return dense<T>(data(v));
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_viewer(details::var_base<T, Allocator>& v) MUDA_NOEXCEPT
{
    return make_dense(v);
}

//print convert
template <typename T>
MUDA_INLINE MUDA_GENERIC const T& printConvert(const dense<T>& idx) MUDA_NOEXCEPT
{
    return idx.operator const T&();
}
}  // namespace muda