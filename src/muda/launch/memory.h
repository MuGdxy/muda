#pragma once
#include <muda/launch/launch_base.h>
#include <muda/tools/version.h>

namespace muda
{
class Memory : public LaunchBase<Memory>
{
  public:
    MUDA_HOST Memory(cudaStream_t stream = nullptr)
        : LaunchBase(stream){};

    template <typename T>
    MUDA_HOST Memory& alloc(T** ptr, size_t byte_size, bool async = DEFAULT_ASYNC_ALLOC_FREE);
    MUDA_HOST Memory& free(void* ptr, bool async = DEFAULT_ASYNC_ALLOC_FREE);

    MUDA_HOST Memory& copy(void* dst, const void* src, size_t byte_size, cudaMemcpyKind kind);
    MUDA_HOST Memory& transfer(void* dst, const void* src, size_t byte_size);
    MUDA_HOST Memory& download(void* dst, const void* src, size_t byte_size);
    MUDA_HOST Memory& upload(void* dst, const void* src, size_t byte_size);

    MUDA_HOST Memory& set(void* data, size_t byte_size, char byte = 0);
};

}  // namespace muda

#include "details/memory.inl"