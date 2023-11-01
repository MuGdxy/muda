#pragma once
#include <muda/compute_graph/compute_graph.h>

namespace muda
{
template <typename F, typename UserTag>
MUDA_HOST ParallelFor& ParallelFor::apply(int count, F&& f, UserTag tag)
{
    using CallableType = raw_type_t<F>;
    static_assert(std::is_invocable_v<CallableType, int>
                      || std::is_invocable_v<CallableType, ParallelForDetails>,
                  "f must be void (int) or void (ParallelForDetails)");

    ComputeGraphBuilder::invoke_phase_actions(
        [&] {  // direct invoke
            invoke(count, std::forward<F>(f), tag);
        },
        [&]
        {
            // as node parms
            auto parms = as_node_parms(count, std::forward<F>(f), tag);
            details::ComputeGraphAccessor().set_kernel_node(parms);
        },
        [&]
        {
            // topo build
            details::ComputeGraphAccessor()
                .set_kernel_node<details::ParallelForCallable<raw_type_t<F>>>(nullptr);
        });
    pop_kernel_name();
    return *this;
}

template <typename F, typename UserTag>
MUDA_HOST void ParallelFor::invoke(int count, F&& f, UserTag tag)
{
    using CallableType = raw_type_t<F>;
    // check_input(count);
    if(count > 0)
    {
        if(m_grid_dim <= 0)  // parallel for
        {
            // calculate the blocks we need
            auto n_blocks = calculate_grid_dim(count);
            auto callable = details::ParallelForCallable<CallableType>{f, count};
            details::parallel_for_kernel<CallableType, UserTag>
                <<<n_blocks, m_block_dim, m_shared_mem_size, m_stream>>>(callable);
        }
        else  // grid stride loop
        {
            auto callable = details::ParallelForCallable<CallableType>{f, count};
            details::grid_stride_loop_kernel<CallableType, UserTag>
                <<<m_grid_dim, m_block_dim, m_shared_mem_size, m_stream>>>(callable);
        }
    }
}

template <typename F, typename UserTag>
MUDA_INLINE auto ParallelFor::as_node_parms(int count, F&& f, UserTag tag)
    -> S<KernelNodeParms<details::ParallelForCallable<raw_type_t<F>>>>
{
    using CallableType = raw_type_t<F>;
    static_assert(std::is_invocable_v<CallableType, int>
                      || std::is_invocable_v<CallableType, ParallelForDetails>,
                  "f must be void (int) or void (ParallelForDetails)");

    check_input(count);

    auto parms =
        std::make_shared<KernelNodeParms<details::ParallelForCallable<CallableType>>>(
            std::forward<F>(f), count);
    if(m_grid_dim <= 0)  // dynamic grid dim
    {

        auto n_blocks = calculate_grid_dim(count);
        parms->func((void*)details::parallel_for_kernel<CallableType, UserTag>);
        parms->grid_dim(n_blocks);
    }
    else  // grid-stride loop
    {
        parms->func((void*)details::grid_stride_loop_kernel<CallableType, UserTag>);
        parms->grid_dim(m_grid_dim);
    }

    parms->block_dim(m_block_dim);
    parms->shared_mem_bytes(m_shared_mem_size);
    parms->parse([](details::ParallelForCallable<CallableType>& p) -> std::vector<void*>
                 { return {&p}; });

    return parms;
}

MUDA_INLINE MUDA_GENERIC int ParallelFor::calculate_grid_dim(int count) const MUDA_NOEXCEPT
{
    auto min_threads = count;
    auto min_blocks  = (min_threads + m_block_dim - 1) / m_block_dim;
    return min_blocks;
}

MUDA_INLINE MUDA_GENERIC void ParallelFor::check_input(int count) const MUDA_NOEXCEPT
{
    MUDA_KERNEL_ASSERT(count >= 0, "count must be >= 0");
    MUDA_KERNEL_ASSERT(m_block_dim > 0, "blockDim must be > 0");
}

MUDA_INLINE MUDA_DEVICE int ParallelForDetails::active_num_in_block() const MUDA_NOEXCEPT
{
    if(m_type == ParallelForType::DynamicBlocks)
    {
        auto block_id = blockIdx.x;
        return (blockIdx.x == gridDim.x - 1) ? m_total_num - block_id * blockDim.x :
                                               blockDim.x;
    }
    else if(m_type == ParallelForType::GridStrideLoop)
    {
        return m_active_num_in_block;
    }
    else
    {
        MUDA_KERNEL_ERROR("invalid paralell for type");
    }
}

MUDA_INLINE MUDA_DEVICE bool ParallelForDetails::is_final_block() const MUDA_NOEXCEPT
{
    if(m_type == ParallelForType::DynamicBlocks)
    {
        return (blockIdx.x == gridDim.x - 1);
    }
    else if(m_type == ParallelForType::GridStrideLoop)
    {
        return m_active_num_in_block == blockDim.x;
    }
    else
    {
        MUDA_KERNEL_ERROR("invalid paralell for type");
    }
}
}  // namespace muda