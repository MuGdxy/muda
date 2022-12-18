#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <exception>

namespace muda
{
inline const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorName(error);
}

template <typename T>
inline void check(T result, char const* const func, const char* const file, int const line)
{
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
}
}  // namespace muda
