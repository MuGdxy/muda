#pragma once
#include <muda/tools/version.h>
#include <vector>
#include <thrust/device_allocator.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <muda/muda_def.h>
#include <muda/viewer/dense.h>

namespace muda
{
namespace details
{
    template <typename T, typename Allocator>
    class VarBase
    {
      public:
        //using T = int;
        //using Allocator = thrust::device_allocator<int>;
        using pointer       = typename Allocator::pointer;
        using const_pointer = typename Allocator::const_pointer;

        MUDA_HOST VarBase() MUDA_NOEXCEPT : m_data(Allocator().allocate(1)) {}

        MUDA_HOST VarBase(const T& value) MUDA_NOEXCEPT : m_data(Allocator().allocate(1))
        {
            this->operator=(value);
        }

        MUDA_HOST ~VarBase() MUDA_NOEXCEPT
        {
            Allocator().deallocate(m_data, 1);
        }

        pointer       data() { return m_data; }
        const_pointer data() const { return m_data; }

        // copy value from host to device
        MUDA_HOST VarBase& operator=(const T& rhs)
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

template <typename T>
class DeviceVar : public details::VarBase<T, thrust::device_allocator<T>>
{
  public:
    using Base = details::VarBase<T, thrust::device_allocator<T>>;
    using Base::Base;
    using Base::operator=;

    const T* data() const { return thrust::raw_pointer_cast(Base::data()); }
    T*       data() { return thrust::raw_pointer_cast(Base::data()); }

    auto viewer() { return Dense<T>(this->data()); }
};
}  // namespace muda


// cast
namespace muda
{
template <typename T, typename Allocator>
MUDA_INLINE const T* data(const details::VarBase<T, Allocator>& v) MUDA_NOEXCEPT
{
    return thrust::raw_pointer_cast(v.data());
}

template <typename T, typename Allocator>
MUDA_INLINE T* data(details::VarBase<T, Allocator>& v) MUDA_NOEXCEPT
{
    return thrust::raw_pointer_cast(v.data());
}
}  // namespace muda


// viewer
namespace muda
{
template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense(details::VarBase<T, Allocator>& v) MUDA_NOEXCEPT
{
    return Dense<T>(::muda::data(v));
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_viewer(details::VarBase<T, Allocator>& v) MUDA_NOEXCEPT
{
    return make_dense(v);
}

//print convert
template <typename T>
MUDA_INLINE MUDA_GENERIC const T& print_convert(const Dense<T>& idx) MUDA_NOEXCEPT
{
    return idx.operator const T&();
}
}  // namespace muda