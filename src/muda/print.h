#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace muda
{
template <typename InType, typename OutType = InType>
inline __host__ __device__ OutType printConvert(const InType& v)
{
    return v;
}

template <typename T>
inline __host__ __device__ const T& printCheck(const T& t)
{
    static_assert(std::is_arithmetic_v<T> || std::is_pointer_v<T>
                      || std::is_same_v<T, std::nullptr_t>
                      || std::is_same_v<T, const char* const>
                      || std::is_same_v<T, const char*>,
                  "not supported type T in printf!");
    return t;
}

template <typename... Args>
inline __host__ __device__ void print(const char* const fmt, Args... arg)
{
    printf(fmt, printCheck(printConvert(arg))...);
}
}  // namespace muda