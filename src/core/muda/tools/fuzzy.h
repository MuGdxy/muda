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

MUDA_THREAD_ONLY inline int memcmp(const char* p1, const char* p2, size_t n)
{
#ifdef __CUDA_ARCH__
    for(; n > 0; ++p1, ++p2, --n)
    {
        if(*p1 != *p2)
            return unchar_t(*p1) < unchar_t(*p2) ? -1 : 1;
    }
    return 0;
#else
    return ::memcmp(p1, p2, n);
#endif
}

MUDA_THREAD_ONLY inline void memmove(char* pDestination, const char* pSource, size_t n)
{
#ifdef __CUDA_ARCH__
    // Create a temporary array to hold data of src
    char* temp = new char[n];

    // Copy data from csrc[] to temp[]
    for(int i = 0; i < n; i++)
        temp[i] = pSource[i];

    // Copy data from temp[] to cdest[]
    for(int i = 0; i < n; i++)
        pDestination[i] = temp[i];

    delete[] temp;
#else
    ::memmove(pDestination, pSource, n);
#endif
}

MUDA_THREAD_ONLY inline void* memset(char* pDestination, int c, size_t n)
{
#ifdef __CUDA_ARCH__
    for(size_t i = 0; i < n; i++)
        pDestination[i] = c;
    return pDestination + n;
#else
    return ::memset(pDestination, c, n);
#endif
}

MUDA_THREAD_ONLY inline char* memcpy(char* pDestination, const char* pSource, size_t n)
{
#ifdef __CUDA_ARCH__
    for(size_t i = 0; i < n; i++)
        pDestination[i] = pSource[i];
    return pDestination + n;
#else
    return (char*)::memcpy(pDestination, pSource, n);
#endif
}
}  // namespace muda