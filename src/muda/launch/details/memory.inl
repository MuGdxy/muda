#pragma once
#include <muda/compute_graph/compute_graph.h>

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
}  // namespace muda