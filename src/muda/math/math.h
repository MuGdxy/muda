#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdlib>
#include <cinttypes>
#include <Eigen/Core>
#include "../muda_def.h"
namespace muda
{
template <typename T>
inline MUDA_GENERIC T abs(T a)
{
    return ::abs(a);
}


template <typename T>
inline MUDA_GENERIC T pow(T a, T b)
{
    return ::pow(a, b);
}

template <typename T>
inline MUDA_GENERIC T sqrt(T a)
{
    return ::sqrt(a);
}

template <typename T>
inline MUDA_GENERIC T cos(T a)
{
    return ::cos(a);
}

template <typename T>
inline MUDA_GENERIC T sin(T a)
{
    return ::sin(a);
}

template <typename T>
inline MUDA_GENERIC T log(T a)
{
    return ::log(a);
}

template <typename T>
inline MUDA_GENERIC T exp(T a)
{
    return ::exp(a);
}

inline MUDA_GENERIC uint32_t next_pow2(uint32_t x)
{
    x -= 1;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x + 1;
}

template <typename T>
inline MUDA_GENERIC const T& max(const T& l, const T& r)
{
    return l > r ? l : r;
}

template <typename T>
inline MUDA_GENERIC const T& min(const T& l, const T& r)
{
    return l < r ? l : r;
}


template <typename T, int N>
inline MUDA_GENERIC const T& max(const Eigen::Vector<T, N>& v)
{
    auto* ret = &v(0);
#pragma unroll
    for(int i = 1; i < N; ++i)
        if(v(i) > v(0))
            ret = &v(i);
    return *ret;
}

template <typename T, int N>
inline MUDA_GENERIC const T& min(const Eigen::Vector<T, N>& v)
{
    auto* ret = &v(0);
#pragma unroll
    for(int i = 1; i < N; ++i)
        if(v(i) < v(0))
            ret = &v(i);
    return *ret;
}

template <typename T>
inline MUDA_GENERIC T clamp(const T& x, const T& l, const T& r)
{
    return min(max(x, l), r);
}

template <typename T>
inline MUDA_GENERIC int signof(const T& x)
{
    if(x > T(0))
        return 1;
    if(x < T(0))
        return -1;
    return 0;
}
}  // namespace muda