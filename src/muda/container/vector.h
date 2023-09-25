#pragma once

#include <muda/tools/version.h>
#include <thrust/device_allocator.h>

#ifdef MUDA_WITH_THRUST_UNIVERSAL
#include <thrust/universal_allocator.h>
#include <thrust/universal_vector.h>
#endif

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include <muda/muda_def.h>
#include <muda/viewer.h>

namespace muda
{
namespace details
{
    template <typename T, typename Alloc>
    using vector_base = thrust::detail::vector_base<T, Alloc>;
}

template <typename T, typename Alloc = thrust::device_allocator<T>>
using DeviceVector = thrust::device_vector<T, Alloc>;

#ifdef MUDA_WITH_THRUST_UNIVERSAL
template <typename T, typename Alloc = thrust::universal_allocator<T>>
using universal_vector = thrust::universal_vector<T, Alloc>;
#endif

template <typename T, typename Alloc = ::std::allocator<T>>
using HostVector = thrust::host_vector<T, Alloc>;
}  // namespace muda

namespace muda
{
template <typename T, typename DevAlloc = thrust::device_allocator<T>, typename HostAlloc = ::std::allocator<T>>
DeviceVector<T, DevAlloc> to_device(const HostVector<T, HostAlloc>& host_vec)
{
    DeviceVector<T, DevAlloc> dev_vec = host_vec;
    return dev_vec;
}

template <typename T, typename DevAlloc = thrust::device_allocator<T>, typename HostAlloc = ::std::allocator<T>>
HostVector<T, HostAlloc> to_host(const DeviceVector<T, DevAlloc>& dev_vec)
{
    HostVector<T, HostAlloc> host_vec = dev_vec;
    return host_vec;
}

// raw pointer
template <typename T, typename Allocator>
MUDA_INLINE const T* data(const details::vector_base<T, Allocator>& v) MUDA_NOEXCEPT
{
    return thrust::raw_pointer_cast(v.data());
}

template <typename T, typename Allocator>
MUDA_INLINE T* data(details::vector_base<T, Allocator>& v) MUDA_NOEXCEPT
{
    return thrust::raw_pointer_cast(v.data());
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense(details::vector_base<T, Allocator>& v) MUDA_NOEXCEPT
{
    return muda::DenseND<T, 1>(::muda::data(v), v.size());
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_viewer(details::vector_base<T, Allocator>& v) MUDA_NOEXCEPT
{
    return make_dense(v);
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense2D(details::vector_base<T, Allocator>& v, int dimy) MUDA_NOEXCEPT
{
    return make_dense2D(::muda::data(v), v.size() / dimy, dimy);
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense2D(details::vector_base<T, Allocator>& v, int dimx, int dimy) MUDA_NOEXCEPT
{
    MUDA_KERNEL_ASSERT(
        dimx * dimy <= v.size(), "dimx=%d, dimy=%d, v.size()=%d\n", dimx, dimy, v.size());
    return make_dense2D(::muda::data(v), dimx, dimy);
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense2D(details::vector_base<T, Allocator>& v,
                                  const ::Eigen::Vector2i& dim) MUDA_NOEXCEPT
{
    MUDA_KERNEL_ASSERT(dim.x() * dim.y() <= v.size(),
                       "dim.x()=%d, dim.y()=%d, v.size()=%d\n",
                       dim.x(),
                       dim.y(),
                       v.size());
    return make_dense2D(::muda::data(v), dim.x(), dim.y());
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense3D(details::vector_base<T, Allocator>& v, int dimy, int dimz) MUDA_NOEXCEPT
{
    MUDA_KERNEL_ASSERT(
        dimy * dimz <= v.size(), "dimy=%d, dimz=%d, v.size()=%d\n", dimy, dimz, v.size());
    return make_dense3D(::muda::data(v), v.size() / (dimy * dimz), dimy, dimz);
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense3D(details::vector_base<T, Allocator>& v,
                                  const ::Eigen::Vector2i dimyz) MUDA_NOEXCEPT
{
    MUDA_KERNEL_ASSERT(dimyz(0) * dimyz(1) <= v.size(),
                       "dimy=%d, dimz=%d, v.size()=%d\n",
                       dimyz(0),
                       dimyz(1),
                       v.size());
    return make_dense3D(::muda::data(v), v.size() / (dimyz(0) * dimyz(1)), dimyz(0), dimyz(1));
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense3D(details::vector_base<T, Allocator>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    MUDA_KERNEL_ASSERT(dimx * dimy * dimz <= v.size(),
                       "dimx=%d, dimy=%d, dimz=%d, v.size()=%d\n",
                       dimx,
                       dimy,
                       dimz,
                       v.size());
    return make_dense3D(::muda::data(v), dimx, dimy, dimz);
}

template <typename T, typename Allocator>
MUDA_INLINE MUDA_HOST auto make_dense3D(details::vector_base<T, Allocator>& v,
                                  const ::Eigen::Vector3i& dim) MUDA_NOEXCEPT
{
    MUDA_KERNEL_ASSERT(dim.x() * dim.y() * dim.z() <= v.size(),
                       "dim.x()=%d, dim.y()=%d, dim.z()=%d, v.size()=%d\n",
                       dim.x(),
                       dim.y(),
                       dim.z(),
                       v.size());
    return make_dense3D(::muda::data(v), dim.x(), dim.y(), dim.z());
}
}  // namespace muda
#include <string>
#include <fstream>

namespace muda
{
///<summary>
///
///</summary>
///<typeparam name="T">value type</typeparam>
///<typeparam name="F">callable object type</typeparam>
///<param name="header"></param>
///<param name="h">host vector</param>
///<param name="filename"></param>
///<param name="ele_in_line">element count in a line</param>
template <typename T, typename F>
MUDA_INLINE void csv(F&& header,  //callable: void (std::ostream& o)
                const HostVector<T>& h,
                const ::std::string&  filename    = "data.csv",
                int                   ele_in_line = 1)
{
    std::ofstream o;
    o.open(filename);
    static_assert(std::is_invocable_v<F, std::ostream&>,
                  "require callable: void (std::ostream& o)");
    header(o);
    for(size_t i = 0; i < h.size(); ++i)
    {
        o << h[i];
        if(i % ele_in_line == 0)
            o << "\n";
        else
            o << ",";
    }
}

/// <summary>
///
/// </summary>
/// <typeparam name="T">value type</typeparam>
/// <param name="h">host vector</param>
/// <param name="filename">filename for saving</param>
/// <param name="ele_in_line">element count in a line</param>
template <typename T>
MUDA_INLINE void csv(const HostVector<T>& h,
                const ::std::string&  filename    = "data.csv",
                int                   ele_in_line = 1)
{
    std::ofstream o;
    o.open(filename);
    for(size_t i = 0; i < h.size(); ++i)
    {
        o << h[i];
        if(i % ele_in_line == 0)
            o << "\n";
        else
            o << ",";
    }
}
}  // namespace muda