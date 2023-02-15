#pragma once
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
namespace muda
{
inline __host__ __device__ void trap()
{
#ifdef __CUDA_ARCH__
    __trap();
#else
    assert(0 && "trap");
#endif
}

inline __host__ __device__ void break_point()
{
#ifdef __CUDA_ARCH__
    __brkpt();
#else
    // TODO:
    // __debugbreak();
#endif
}
}  // namespace muda