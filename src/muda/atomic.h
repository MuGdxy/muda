//#pragma once
//#include <muda/muda_def.h>
//#include <muda/cuda/device_functions.h>
//#include <Eigen/Core>
//
//namespace muda
//{
//template <typename T, int N>
//MUDA_INLINE MUDA_GENERIC Eigen::Vector<T, N> atomicAdd(Eigen::Vector<T, N>& v,
//                                                       const Eigen::Vector<T, N>& val) MUDA_NOEXCEPT
//{
//    Eigen::Vector<T, N> ret;
//#pragma unroll
//    for(int i = 0; i < N; ++i)
//        ret(i) = atomicAdd(&v(i), val(i));
//    return ret;
//}
//
//template <typename T, int M, int N>
//MUDA_INLINE MUDA_GENERIC Eigen::Matrix<T, M, N> atomicAdd(Eigen::Matrix<T, M, N>& v,
//                                                          const Eigen::Matrix<T, M, N>& val) MUDA_NOEXCEPT
//{
//    Eigen::Matrix<T, M, N> ret;
//#pragma unroll
//    for(int i = 0; i < M; ++i)
//#pragma unroll
//        for(int j = 0; j < N; ++j)
//            ret(i, j) = atomicAdd(&v(i, j), val(i, j));
//    return ret;
//}
//}  // namespace muda