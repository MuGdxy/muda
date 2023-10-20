#pragma once
#include <muda/compute_graph/compute_graph.h>
#include "memory.h"

namespace muda
{
MUDA_INLINE Memory& Memory::copy(void* dst, const void* src, size_t byte_size, cudaMemcpyKind kind)
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
MUDA_INLINE Memory& Memory::set(void* data, size_t byte_size, char byte)
{
    ComputeGraphBuilder::invoke_phase_actions(
        [&] {
            checkCudaErrors(cudaMemsetAsync(data, (int)byte, byte_size, stream()));
        },
        [&] { MUDA_ERROR_WITH_LOCATION("not implemented"); });
    return *this;
}

}  // namespace muda