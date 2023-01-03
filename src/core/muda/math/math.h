#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdlib>
#include <cinttypes>
#include <Eigen/Core>
#include <Eigen/SVD>

#include "cuda_svd.h"
#include "../muda_def.h"
#undef max
#undef min
namespace muda
{
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

template <typename T, int Size>
inline MUDA_GENERIC Eigen::Vector<T, Size> Max(const Eigen::Vector<T, Size>& l,
                                               const Eigen::Vector<T, Size>& r)
{
    Eigen::Vector<T, Size> max;
#pragma unroll
    for(size_t i = 0; i < Size; ++i)
        max(i) = muda::max(l(i), r(i));
    return max;
}

template <typename T, int Size>
inline MUDA_GENERIC Eigen::Vector<T, Size> Min(const Eigen::Vector<T, Size>& l,
                                        const Eigen::Vector<T, Size>& r)
{
    Eigen::Vector<T, Size> min;
#pragma unroll
    for(size_t i = 0; i < Size; ++i)
        min(i) = muda::min(l(i), r(i));
    return min;
}

template <typename T, int N>
inline MUDA_GENERIC T Dot(const Eigen::Vector<T, N>& x, const Eigen::Vector<T, N>& y)
{
    return x.dot(y);
}

template <typename T>
inline MUDA_GENERIC Eigen::Vector3<T> Cross(const Eigen::Vector3<T>& x,
                                     const Eigen::Vector3<T>& y)
{
    return x.cross(y);
    //return Eigen::Vector3<T>(x.y() * y.z() - y.y() * x.z(),
    //                         x.z() * y.x() - y.z() * x.x(),
    //                         x.x() * y.y() - y.x() * x.y());
}

template <typename T>
inline MUDA_GENERIC T MixProduct(const Eigen::Vector3<T>& x,
                          const Eigen::Vector3<T>& y,
                          const Eigen::Vector3<T>& z)
{
    return Dot(Cross(x, y), z);
}

template <typename T>
inline MUDA_GENERIC void SVD(const Eigen::Matrix<T, 3, 3>& F,
                      Eigen::Matrix<T, 3, 3>&       U,
                      Eigen::Vector3<T>&            Sigma,
                      Eigen::Matrix<T, 3, 3>&       V)
{
    using mat3 = Eigen::Matrix<T, 3, 3>;
    using vec3 = Eigen::Vector3<T>;
#ifdef __CUDA_ARCH__
    static_assert(std::is_same_v<T, float>, "only allow float on gpu");
    details::svd(F(0, 0),
                 F(0, 1),
                 F(0, 2),
                 F(1, 0),
                 F(1, 1),
                 F(1, 2),
                 F(2, 0),
                 F(2, 1),
                 F(2, 2),
                 U(0, 0),
                 U(0, 1),
                 U(0, 2),
                 U(1, 0),
                 U(1, 1),
                 U(1, 2),
                 U(2, 0),
                 U(2, 1),
                 U(2, 2),
                 Sigma(0),
                 Sigma(1),
                 Sigma(2),
                 V(0, 0),
                 V(0, 1),
                 V(0, 2),
                 V(1, 0),
                 V(1, 1),
                 V(1, 2),
                 V(2, 0),
                 V(2, 1),
                 V(2, 2));
#else
    const Eigen::JacobiSVD<mat3, Eigen::NoQRPreconditioner> svd(
        F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U     = svd.matrixU();
    V     = svd.matrixV();
    Sigma = svd.singularValues();
#endif
    mat3 L  = mat3::Identity();
    L(2, 2) = (U * V.transpose()).determinant();

    const T detU = U.determinant();
    const T detV = V.determinant();

    if(detU < 0.0 && detV > 0)
        U = U * L;
    if(detU > 0.0 && detV < 0.0)
        V = V * L;
    Sigma[2] = Sigma[2] * L(2, 2);
}

template <typename T>
inline MUDA_GENERIC void PolarDecomp(const Eigen::Matrix<T, 3, 3>& F,
                              Eigen::Matrix<T, 3, 3>&       R,
                              Eigen::Matrix<T, 3, 3>&       S)
{
    Eigen::Matrix<T, 3, 3> U, V;
    Eigen::Vector3<T>      Sigma;
    SVD(F, U, Sigma, V);
    R = U * V.transpose();
    S = V * Sigma.asDiagonal() * V.transpose();
}
}  // namespace muda