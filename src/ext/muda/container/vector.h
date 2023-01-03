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

/// <summary>
/// make a mapper for all device_vectors, require each device_vector has a larger size than the mapper size
/// later we can use mapper to create idxer in kernel for safe memory access
/// </summary>
/// <typeparam name="...Ts"></typeparam>
/// <typeparam name="N"></typeparam>
/// <param name="dim"></param>
/// <param name="...v"></param>
/// <returns></returns>
template <int N, typename... Vectors>
inline __host__ auto make_mapper(const Eigen::Vector<int, N>& dim, Vectors&... v)
{
    int size = 1;
    for(int i = 0; i < N; ++i)
    {
        auto d = dim[i];
        if(d <= 0)
            throw(std::runtime_error("demension <= 0"));
        size *= d;
    }
    auto sizes = {v.size()...};
    auto min   = std::min(sizes);
    if(size > min)
        throw(std::runtime_error("min(device_vector.size()) is less than the mapper size"));
    return muda::mapper<N>(dim);
}

/// <summary>
/// make a mapper1D for all device_vectors, choose min(device_vector.size()) as the mapper size automatically.
/// later we can use mapper to create idxer in kernel for safe memory access
/// </summary>
/// <typeparam name="...Ts"></typeparam>
/// <param name="...v"></param>
/// <returns></returns>
template <typename... Vectors>
inline __host__ auto make_mapper(Vectors&... v)
{
    auto sizes = {v.size()...};
    auto min   = std::min(sizes);
    if(min <= 0)
        throw(std::runtime_error("some device_vector's size <= 0"));
    return muda::mapper<1>(min);
}

/// <summary>
/// make a mapper1D for all device_vectors, require each device_vector has a larger size than the mapper size
/// later we can use mapper to create idxer in kernel for safe memory access
/// </summary>
/// <typeparam name="...Ts"></typeparam>
/// <param name="...v"></param>
/// <returns></returns>
template <typename... Vectors>
inline __host__ auto make_mapper(int size, Vectors&... v) noexcept
{
    return make_mapper<1>(Eigen::Vector<int, 1>(size), v...);
}

template <int N, typename... Ts>
inline __host__ auto make_idxer(const muda::mapper<N>& m, Ts*... ptr) noexcept
{
    return std::make_tuple(muda::idxerND<Ts, N>(ptr, m)...);
}

template <int N, typename T, typename Allocator>
inline __host__ auto make_idxer(const muda::mapper<N>&              m,
                                details::vector_base<T, Allocator>& v) noexcept
{
    return muda::idxerND<T, N>(data(v), m);
}


template <typename T, typename Allocator>
inline __host__ auto make_idxer(details::vector_base<T, Allocator>& v) noexcept
{
    return muda::idxerND<T, 1>(data(v), v.size());
}

template <typename T, typename Allocator>
inline __host__ auto make_viewer(details::vector_base<T, Allocator>& v) noexcept
{
    return make_idxer(v);
}


template <int N, typename T>
inline __host__ __device__ auto make_idxer(const muda::mapper<N>& m, T* ptr) noexcept
{
    return muda::idxerND<T, N>(ptr, m);
}

template <typename T>
inline __host__ __device__ auto make_idxer(T* ptr, size_t count) noexcept
{
    return muda::idxerND<T, 1>(ptr, count);
}

template <typename T>
inline __host__ __device__ auto make_viewer(T* ptr, size_t count) noexcept
{
    return make_idxer(ptr, count);
}
}  // namespace muda
#include <string>
#include <fstream>

namespace muda
{
template <typename T,typename F>
inline void csv(F&& header, const host_vector<T>& h, const std::string& filename = "data.csv", int ele_in_line = 1)
{
	
    std::ofstream o;
    o.open(filename);
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