#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/compute_graph/nodes/compute_graph_kernel_node.h>
#include <muda/launch/kernel.h>
namespace muda
{
template <typename F, typename UserTag>
MUDA_INLINE MUDA_HOST Launch::S<KernelNodeParms<raw_type_t<F>>> Launch::as_node_parms(F&& f, UserTag tag)
{
    using CallableType = raw_type_t<F>;
    static_assert(std::is_invocable_v<CallableType>, "f:void (void)");
    auto parms = std::make_shared<KernelNodeParms<CallableType>>(std::forward<F>(f));

    parms->func((void*)details::generic_kernel<CallableType, UserTag>);
    parms->grid_dim(m_grid_dim);
    parms->block_dim(m_block_dim);
    parms->shared_mem_bytes(m_shared_mem_size);
    parms->parse([](CallableType& p) -> std::vector<void*> { return {&p}; });
    return parms;
}

template <typename F, typename UserTag>
MUDA_INLINE MUDA_HOST void Launch::invoke(F&& f, UserTag tag)
{
    using CallableType = raw_type_t<F>;
    details::generic_kernel<CallableType, UserTag>
        <<<m_grid_dim, m_block_dim, m_shared_mem_size, m_stream>>>(f);
}

template <typename F, typename UserTag>
MUDA_INLINE MUDA_HOST Launch& Launch::apply(F&& f, UserTag tag)
{
    using CallableType = raw_type_t<F>;
    static_assert(std::is_invocable_v<CallableType>, "f:void (void)");

    ComputeGraphBuilder::invoke_phase_actions(
        [&] { invoke(std::forward<F>(f), tag); },
        [&]
        {
            auto parms = this->as_node_parms(std::forward<F>(f), tag);
            details::ComputeGraphAccessor().set_kernel_node(parms);
        },
        [&]
        {
            details::ComputeGraphAccessor().set_kernel_node<KernelNodeParms<CallableType>>(nullptr);
        });
    pop_kernel_name();

    return *this;
}
}  // namespace muda