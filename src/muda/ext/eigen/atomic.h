#pragma once
#include <muda/atomic.h>
#include <Eigen/Core>

namespace muda::eigen
{
template <typename T, int M, int N>
MUDA_GENERIC Eigen::Matrix<T, M, N> atomic_add(Eigen::Matrix<T, M, N>& dst,
                                               const Eigen::Matrix<T, M, N>& src)
{
    Eigen::Matrix<T, M, N> ret;

#pragma unroll
    for(int j = 0; j < N; ++j)
#pragma unroll
        for(int i = 0; i < M; ++i)
        {
            ret(i, j) = muda::atomic_add(&dst(i, j), src(i, j));
        }
}

template <typename T, int M, int N>
MUDA_GENERIC Eigen::Matrix<T, M, N> atomic_add(Eigen::Map<Eigen::Matrix<T, M, N>>& dst,
                                               const Eigen::Matrix<T, M, N>& src)
{
    Eigen::Matrix<T, M, N> ret;

#pragma unroll
    for(int j = 0; j < N; ++j)
#pragma unroll
        for(int i = 0; i < M; ++i)
        {
            ret(i, j) = muda::atomic_add(&dst(i, j), src(i, j));
        }
}
}  // namespace muda::eigen