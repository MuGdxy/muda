#pragma once
#include <thrust/device_allocator.h>
#include <thrust/universal_allocator.h>
#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <muda/viewer.h>

namespace muda
{
namespace details
{
    template <typename T, typename Alloc>
    using vector_base = thrust::detail::vector_base<T, Alloc>;
}

template <typename T, typename Alloc = thrust::device_allocator<T>>
using device_vector = thrust::device_vector<T, Alloc>;

template <typename T, typename Alloc = thrust::universal_allocator<T>>
using universal_vector = thrust::universal_vector<T, Alloc>;

template <typename T, typename Alloc = std::allocator<T>>
using host_vector = thrust::host_vector<T, Alloc>;
}  // namespace muda

namespace muda
{
template <typename T, typename DevAlloc = thrust::device_allocator<T>, typename HostAlloc = std::allocator<T>>
device_vector<T, DevAlloc> to_device(const host_vector<T, HostAlloc>& host_vec)
{
    device_vector<T, DevAlloc> dev_vec = host_vec;
    return dev_vec;
}

template <typename T, typename DevAlloc = thrust::device_allocator<T>, typename HostAlloc = std::allocator<T>>
host_vector<T, HostAlloc> to_host(const device_vector<T, DevAlloc>& dev_vec)
{
    host_vector<T, HostAlloc> host_vec = dev_vec;
    return host_vec;
}

// raw pointer
template <typename T, typename Allocator>
inline const T* data(const details::vector_base<T, Allocator>& v) noexcept
{
    return thrust::raw_pointer_cast(v.data());
}

template <typename T, typename Allocator>
inline T* data(details::vector_base<T, Allocator>& v) noexcept
{
    return thrust::raw_pointer_cast(v.data());
}

template <typename T, typename Allocator>
inline __host__ auto make_dense(details::vector_base<T, Allocator>& v) noexcept
{
    return muda::denseND<T, 1>(data(v), v.size());
}

template <typename T, typename Allocator>
inline __host__ auto make_viewer(details::vector_base<T, Allocator>& v) noexcept
{
    return make_dense(v);
}

template <typename T>
inline __host__ __device__ auto make_dense(T* ptr, size_t count) noexcept
{
    return muda::dense1D<T>(ptr, count);
}

template <typename T>
inline __host__ __device__ auto make_viewer(T* ptr, size_t count) noexcept
{
    return muda::dense1D<T>(ptr, count);
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
inline void csv(F&& header,  //callable: void (std::ostream& o)
                const host_vector<T>& h,
                const std::string&    filename    = "data.csv",
                int                   ele_in_line = 1)
{
    std::ofstream o;
    o.open(filename);
    static_assert(std::is_invocable_v<F, std::ostream&>, "require callable: void (std::ostream& o)");
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
inline void csv(const host_vector<T>& h, const std::string& filename = "data.csv", int ele_in_line = 1)
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