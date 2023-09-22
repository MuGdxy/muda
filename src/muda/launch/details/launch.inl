#pragma once
#include <muda/compute_graph/compute_graph_builder.h>
namespace muda
{
template <typename F, typename UserTag>
MUDA_INLINE Launch& Launch::apply(F&& f, UserTag tag)
{
    ComputeGraphBuilder::invoke(
        [&] // none
        {
            using CallableType = raw_type_t<F>;
            static_assert(std::is_invocable_v<CallableType>, "f:void (void)");
            details::generic_kernel<CallableType, UserTag>
                <<<m_gridDim, m_block_dim, m_shared_mem_size, m_stream>>>(f);
            finish_kernel_launch();
        },
        [&] // building 
        { 
            auto parms = this->as_node_parms(std::forward<F>(f), tag);

        },
        [&] // updating
        {

        });
    return *this;
}
}  // namespace muda