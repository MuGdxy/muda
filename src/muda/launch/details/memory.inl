#pragma once
#include <muda/compute_graph/compute_graph.h>
namespace muda
{
template <typename T>
MUDA_HOST Memory& Memory::alloc(T** ptr, size_t byte_size, bool async)
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

MUDA_INLINE MUDA_HOST Memory& Memory::free(void* ptr, bool async)
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

MUDA_INLINE MUDA_HOST Memory& Memory::copy(void* dst, const void* src, size_t byte_size, cudaMemcpyKind kind)
{
    ComputeGraphBuilder::invoke_phase_actions(
        [&] {
            checkCudaErrors(cudaMemcpyAsync(dst, src, byte_size, kind, stream()));
        },
        [&]
        {
            details::ComputeGraphAccessor().set_memcpy_node(dst, src, byte_size, kind);
        });
    return *this;
}

MUDA_INLINE MUDA_HOST Memory& Memory::transfer(void* dst, const void* src, size_t byte_size)
{
    return copy(dst, src, byte_size, cudaMemcpyDeviceToDevice);
}

MUDA_INLINE MUDA_HOST Memory& Memory::download(void* dst, const void* src, size_t byte_size)
{
    return copy(dst, src, byte_size, cudaMemcpyDeviceToHost);
}

MUDA_INLINE MUDA_HOST Memory& Memory::upload(void* dst, const void* src, size_t byte_size)
{
    return copy(dst, src, byte_size, cudaMemcpyHostToDevice);
}

MUDA_INLINE MUDA_HOST Memory& Memory::set(void* data, size_t byte_size, char byte)
{
    ComputeGraphBuilder::invoke_phase_actions(
        [&] {
            checkCudaErrors(cudaMemsetAsync(data, (int)byte, byte_size, stream()));
        },
        [&] { MUDA_ERROR_WITH_LOCATION("not implemented"); });
    return *this;
}
}  // namespace muda