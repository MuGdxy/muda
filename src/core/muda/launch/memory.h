#pragma once
#include "launch_base.h"

namespace muda
{
class memory : public launch_base<memory>
{
  public:
    memory(cudaStream_t stream = nullptr)
        : launch_base(stream){};

    template <typename T>
    memory& alloc(T** ptr, size_t byte_size)
    {
        checkCudaErrors(cudaMallocAsync(ptr, byte_size, stream_));
        return *this;
    }

    memory& free(void* ptr)
    {
        checkCudaErrors(cudaFreeAsync(ptr, stream_));
        return *this;
    }

    memory& copy(void* dst, const void* src, size_t byte_size, cudaMemcpyKind kind)
    {
        checkCudaErrors(cudaMemcpyAsync(dst, src, byte_size, kind, stream_));
        return *this;
    }

    template <typename T>
    [[nodiscard]] static auto asAllocNodeParms(size_t count)
    {
        auto parms = std::make_shared<memAllocNodeParms<T>>(count);
        return parms;
    }

    memory& set(void* data, size_t byte_size, char byte = 0)
    {
        checkCudaErrors(cudaMemsetAsync(data, (int)byte, byte_size, stream_));
        return *this;
    }
};
}  // namespace muda