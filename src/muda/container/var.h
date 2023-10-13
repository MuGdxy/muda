#pragma once
//#include <muda/tools/version.h>
//#include <vector>
//#include <thrust/device_allocator.h>
//
//#include <thrust/detail/raw_pointer_cast.h>
//#include <thrust/fill.h>
//#include <thrust/copy.h>
//#include <muda/muda_def.h>
//#include <muda/buffer/var_view.h>
//#include <muda/viewer/dense.h>
//
//namespace muda
//{
//namespace details
//{
//    template <typename T, typename Allocator>
//    class VarBase
//    {
//      public:
//        //using T = int;
//        //using Allocator = thrust::device_allocator<int>;
//        using pointer       = typename Allocator::pointer;
//        using const_pointer = typename Allocator::const_pointer;
//
//        VarBase() MUDA_NOEXCEPT : m_data(Allocator().allocate(1)) {}
//
//        VarBase(const T& value) MUDA_NOEXCEPT : m_data(Allocator().allocate(1))
//        {
//            this->operator=(value);
//        }
//
//        ~VarBase() MUDA_NOEXCEPT { Allocator().deallocate(m_data, 1); }
//
//        pointer       data() { return m_data; }
//        const_pointer data() const { return m_data; }
//
//        // copy value from host to device
//        VarBase& operator=(const T& rhs)
//        {
//            thrust::fill_n(data(), 1, rhs);
//            return *this;
//        }
//
//        // copy value from device to host
//        operator T()
//        {
//            T t;
//            thrust::copy_n(data(), 1, &t);
//            return t;
//        }
//
//      private:
//        pointer m_data;
//    };
//}  // namespace details
//
//template <typename T>
//class DeviceVar : public details::VarBase<T, thrust::device_allocator<T>>
//{
//  public:
//    using Base = details::VarBase<T, thrust::device_allocator<T>>;
//    using Base::Base;
//    using Base::operator=;
//
//    const T* data() const { return thrust::raw_pointer_cast(Base::data()); }
//    T*       data() { return thrust::raw_pointer_cast(Base::data()); }
//
//    DeviceVar& operator=(DeviceVar<T> other)
//    {
//        view().copy_from(other.view());
//        return *this;
//    }
//
//    DeviceVar& operator=(VarView other)
//    {
//        view().copy_from(other);
//        return *this;
//    }
//
//    auto viewer() { return Dense<T>(this->data()); }
//    auto cviewer() const { return CDense<T>(this->data()); }
//    auto view() const { return VarView<T>{m_data}; }
//    operator VarView<T>() { return view(); }
//};
//}  // namespace muda

#include <muda/buffer/device_var.h>