#pragma once
#include "launch_base.h"
#include <muda/tools/version.h>

namespace muda
{
class Memory : public LaunchBase<Memory>
{
  public:
    Memory(cudaStream_t stream = nullptr)
        : LaunchBase(stream){};

    template <typename T>
    Memory& alloc(T** ptr, size_t byte_size, bool async = DEFAULT_ASYNC_ALLOC_FREE)
    {

#ifdef MUDA_WITH_ASYNC_MEMORY_ALLOC_FREE
        if(async)
            checkCudaErrors(cudaMallocAsync(ptr, byte_size, stream()));
        else
            checkCudaErrors(cudaMalloc(ptr, byte_size));
#else
        checkCudaErrors(cudaMalloc(ptr, byte_size));
#endif
        return *this;
    }

    Memory& free(void* ptr, bool async = DEFAULT_ASYNC_ALLOC_FREE)
    {
#ifdef MUDA_WITH_ASYNC_MEMORY_ALLOC_FREE
        if(async)
            checkCudaErrors(cudaFreeAsync(ptr, stream()));
        else
            checkCudaErrors(cudaFree(ptr));
#else
        checkCudaErrors(cudaFree(ptr));
#endif
        return *this;
    }

    Memory& copy(void* dst, const void* src, size_t byte_size, cudaMemcpyKind kind);

    Memory& transfer(void* dst, const void* src, size_t byte_size)
    {
        return copy(dst, src, byte_size, cudaMemcpyDeviceToDevice);
    }

    Memory& download(void* dst, const void* src, size_t byte_size)
    {
        return copy(dst, src, byte_size, cudaMemcpyDeviceToHost);
    }

    Memory& upload(void* dst, const void* src, size_t byte_size)
    {
        copy(dst, src, byte_size, cudaMemcpyHostToDevice);
        return *this;
    }

    Memory& set(void* data, size_t byte_size, char byte = 0)
    {
        checkCudaErrors(cudaMemsetAsync(data, (int)byte, byte_size, stream()));
        return *this;
    }
};
}  // namespace muda

#include <muda/launch/details/memory.inl>