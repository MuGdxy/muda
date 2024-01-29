#pragma once
#include <muda/muda_def.h>
#include <vector_types.h>
#include <math_constants.h>

namespace muda::lbvh
{
template <typename T>
struct vector_of;
template <>
struct vector_of<float>
{
    using type = float4;
};
template <>
struct vector_of<double>
{
    using type = double4;
};

template <typename T>
using vector_of_t = typename vector_of<T>::type;

template <typename T>
__device__ inline T infinity() noexcept;

template <>
__device__ inline float infinity<float>() noexcept
{
    return CUDART_INF_F;
}
template <>
__device__ inline double infinity<double>() noexcept
{
    return CUDART_INF;
}

}  // namespace muda::lbvh
