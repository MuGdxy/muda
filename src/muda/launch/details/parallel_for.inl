#pragma once
#include <muda/compute_graph/compute_graph.h>

namespace muda
{
template <typename F, typename UserTag>
MUDA_INLINE ParallelFor& ParallelFor::apply(int count, F&& f, UserTag tag)
{
    using CallableType = raw_type_t<F>;
    static_assert(std::is_invocable_v<CallableType, int>
                      || std::is_invocable_v<CallableType, int, ParallelForDetails>,
                  "f must be void (int) or void (int, ParallelForDetails)");

    check_input(count);

    ComputeGraphBuilder::invoke_phase_actions(
        [&] { invoke(count, std::forward<F>(f), tag); },
        [&]
        {
            auto parms = as_node_parms(count, std::forward<F>(f), tag);
            details::ComputeGraphAccessor().set_kernel_node(parms);
        });
    return *this;
}

template <typename F, typename UserTag>
MUDA_INLINE void ParallelFor::invoke(int count, F&& f, UserTag tag)
{
    using CallableType = raw_type_t<F>;

    if(m_gridDim <= 0)  // parallel for
    {
        if(count > 0)
        {
            // calculate the blocks we need
            auto n_blocks = calculate_grid_dim(count);
            if constexpr(std::is_invocable_v<CallableType, int>)
            {
                details::parallel_for_kernel<CallableType, UserTag>
                    <<<n_blocks, m_block_dim, m_shared_mem_size, this->stream()>>>(f, count);
            }
            else if constexpr(std::is_invocable_v<CallableType, int, ParallelForDetails>)
            {
                details::parallel_for_kernel_with_details<CallableType, UserTag>
                    <<<n_blocks, m_block_dim, m_shared_mem_size, this->stream()>>>(f, count);
            }
            else
            {
                static_assert("f must be void (int) or void (int, ParallelForDetails)");
            }
        }
    }
    else  // grid stride loop
    {
        if constexpr(std::is_invocable_v<CallableType, int>)
        {
            details::grid_stride_loop_kernel<CallableType, UserTag>
                <<<m_gridDim, m_block_dim, m_shared_mem_size, this->stream()>>>(f, count);
        }
        else if constexpr(std::is_invocable_v<CallableType, int, ParallelForDetails>)
        {
            details::grid_stride_loop_kernel_with_details<CallableType, UserTag>
                <<<m_gridDim, m_block_dim, m_shared_mem_size, this->stream()>>>(f, count);
        }
        else
        {
            static_assert("f must be void (int) or void (int, ParallelForDetails)");
        }
    }
    finish_kernel_launch();
}

template <typename F, typename UserTag>
MUDA_INLINE auto ParallelFor::as_node_parms(int count, F&& f, UserTag tag)
    -> S<KernelNodeParms<KernelData<raw_type_t<F>>>>
{
    using CallableType = raw_type_t<F>;
    static_assert(std::is_invocable_v<CallableType, int>
                      || std::is_invocable_v<CallableType, int, ParallelForDetails>,
                  "f must be void (int) or void (int, ParallelForDetails)");

    check_input(count);

    auto parms = std::make_shared<KernelNodeParms<KernelData<CallableType>>>(
        count, std::forward<F>(f));
    if(m_gridDim <= 0)  // dynamic grid dim
    {
        if constexpr(std::is_invocable_v<CallableType, int>)
        {
            auto n_blocks = calculate_grid_dim(count);
            parms->func((void*)details::parallel_for_kernel<CallableType, UserTag>);
            parms->gridDim(n_blocks);
        }
        else if constexpr(std::is_invocable_v<CallableType, int, ParallelForDetails>)
        {
            auto n_blocks = calculate_grid_dim(count);
            parms->func((void*)details::parallel_for_kernel_with_details<CallableType, UserTag>);
            parms->gridDim(n_blocks);
        }
        else
        {
            static_assert("f must be void (int) or void (int, ParallelForDetails)");
        }
    }
    else  // grid-stride loop
    {
        if constexpr(std::is_invocable_v<CallableType, int>)
        {
            parms->func((void*)details::grid_stride_loop_kernel<CallableType, UserTag>);
            parms->gridDim(m_gridDim);
        }
        else if constexpr(std::is_invocable_v<CallableType, int, ParallelForDetails>)
        {
            parms->func((void*)details::grid_stride_loop_kernel_with_details<CallableType, UserTag>);
            parms->gridDim(m_gridDim);
        }
        else
        {
            static_assert("f must be void (int) or void (int, ParallelForDetails)");
        }
    }

    parms->blockDim(m_block_dim);
    parms->sharedMemBytes(m_shared_mem_size);
    parms->parse(
        [](KernelData<CallableType>& p) -> std::vector<void*> {
            return {&p.callable, &p.count};
        });
    finish_kernel_launch();
    return parms;
}

MUDA_INLINE int ParallelFor::calculate_grid_dim(int count) const MUDA_NOEXCEPT
{
    auto min_threads = count;
    auto min_blocks  = (min_threads + m_block_dim - 1) / m_block_dim;
    return min_blocks;
}

MUDA_INLINE void ParallelFor::check_input(int count) const MUDA_NOEXCEPT
{
    MUDA_ASSERT(count >= 0, "count must be >= 0");
    MUDA_ASSERT(m_block_dim > 0, "blockDim must be > 0");
}

MUDA_INLINE int ParallelForDetails::active_num_in_block() const MUDA_NOEXCEPT
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

MUDA_INLINE bool ParallelForDetails::is_final_block() const MUDA_NOEXCEPT
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