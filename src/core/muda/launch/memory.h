#pragma once
#include "launch_base.h"
#include <muda/tools/version.h>

namespace muda
{
class memory : public launch_base<memory>
{
  public:
    memory(cudaStream_t stream = nullptr)
        : launch_base(stream){};

    template <typename T>
    memory& alloc(T** ptr, size_t byte_size, bool async = DEFAULT_ASYNC_ALLOC_FREE)
    {

#ifdef MUDA_WITH_ASYNC_MEMORY_ALLOC_FREE
        if(async)
            checkCudaErrors(cudaMallocAsync(ptr, byte_size, m_stream));
        else
            checkCudaErrors(cudaMalloc(ptr, byte_size));
#else
        checkCudaErrors(cudaMalloc(ptr, byte_size));
#endif
        return *this;
    }

    memory& free(void* ptr, bool async = DEFAULT_ASYNC_ALLOC_FREE)
    {
#ifdef MUDA_WITH_ASYNC_MEMORY_ALLOC_FREE
        if(async)
            checkCudaErrors(cudaFreeAsync(ptr, m_stream));
        else
            checkCudaErrors(cudaFree(ptr));
#else
        checkCudaErrors(cudaFree(ptr));
#endif
        return *this;
    }


#ifdef MUDA_WITH_GRAPH_MEMORY_ALLOC_FREE
    template <typename T>
    MUDA_NODISCARD static auto asAllocNodeParms(size_t count)
    {
        auto parms = std::make_shared<memAllocNodeParms<T>>(count);
        return parms;
    }
#endif

    memory& copy(void* dst, const void* src, size_t byte_size, cudaMemcpyKind kind)
    {
        checkCudaErrors(cudaMemcpyAsync(dst, src, byte_size, kind, m_stream));
        return *this;
    }

    memory& download(void* dst, const void* src, size_t byte_size)
    {
        return copy(dst, src, byte_size, cudaMemcpyDeviceToHost);
    }
    
    memory& upload(void* dst, const void* src, size_t byte_size)
    {
        copy(dst, src, byte_size, cudaMemcpyHostToDevice);
        return *this;
    }

    memory& set(void* data, size_t byte_size, char byte = 0)
    {
        checkCudaErrors(cudaMemsetAsync(data, (int)byte, byte_size, m_stream));
        return *this;
    }
};
}  // namespace muda