#pragma once
#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/compute_graph/nodes/compute_graph_kernel_node.h>
namespace muda
{
template <typename F, typename UserTag>
MUDA_INLINE Launch& Launch::apply(F&& f, UserTag tag)
{
    ComputeGraphBuilder::invoke(
        [&]  // none
        {
            using CallableType = raw_type_t<F>;
            static_assert(std::is_invocable_v<CallableType>, "f:void (void)");
            details::generic_kernel<CallableType, UserTag>
                <<<m_gridDim, m_block_dim, m_shared_mem_size, m_stream>>>(f);
            finish_kernel_launch();
        },
        [&]  // building
        {
            auto parms = this->as_node_parms(std::forward<F>(f), tag);
            details::ComputeGraphAccessor().add_kernel_node(parms);
        },
        [&]  // updating
        {
            auto parms = this->as_node_parms(std::forward<F>(f), tag);
            details::ComputeGraphAccessor().update_kernel_node(parms);
        });
    return *this;
}
}  // namespace muda