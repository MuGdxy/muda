#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace muda
{
__forceinline__ __host__ __device__ dim3 grid_dim()
{
#ifdef __CUDA_ARCH__
    return gridDim;
#else
    return dim3(0, 0, 0);
#endif
}

__forceinline__ __host__ __device__ dim3 block_idx()
{
#ifdef __CUDA_ARCH__
    return blockIdx;
#else
    return dim3(0, 0, 0);
#endif
}

__forceinline__ __host__ __device__ dim3 block_dim()
{
#ifdef __CUDA_ARCH__
    return blockDim;
#else
    return dim3(0, 0, 0);
#endif
}

__forceinline__ __host__ __device__ dim3 thread_idx()
{
#ifdef __CUDA_ARCH__
    return threadIdx;
#else
    return dim3(0, 0, 0);
#endif
}
}  // namespace muda