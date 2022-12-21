#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <exception>

namespace muda
{
__host__ __device__ inline const char* _cudaGetErrorEnum(cudaError_t error)
{
#ifdef __CUDA_ARCH__
    return "<muda: not impl yet>";
#else
    return cudaGetErrorName(error);
#endif
}

template <typename T>
__host__ __device__ inline void check(T                 result,
                                      char const* const func,
                                      const char* const file,
                                      int const         line)
{
#ifdef __CUDA_ARCH__
    if(result)
    {
        printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n",
               file,
               line,
               static_cast<unsigned int>(result),
               _cudaGetErrorEnum(result),
               func);
        if constexpr(trapOnError)
            trap();
    }
#else
    if(result)
    {
        std::fprintf(stderr,
                     "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                     file,
                     line,
                     static_cast<unsigned int>(result),
                     _cudaGetErrorEnum(result),
                     func);
        throw std::exception(_cudaGetErrorEnum(result));
    }
#endif
}
}  // namespace muda
