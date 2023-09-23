#pragma once
#include <muda/compute_graph/compute_graph_node.h>
#include <muda/graph/graph.h>
#include <muda/graph/kernel_node.h>

namespace muda
{
class ComputeGraphKernelNode : public ComputeGraphNodeBase
{
    template <typename T>
    using S = std::shared_ptr<T>;

  protected:
    friend class ComputeGraph;
    friend class details::ComputeGraphAccessor;
    ComputeGraphKernelNode(ComputeGraph*                           graph,
                           std::string_view                        name,
                           NodeId                                  node_id,
                           std::map<VarId, ComputeGraphVarUsage>&& usages)
        : ComputeGraphNodeBase(
            graph, name, node_id, ComputeGraphNodeType::KernelNode, std::move(usages))
    {
    }

    S<KernelNode> m_node;
    void          set_node(S<KernelNode> node)
    {
        m_node = node;
        set_handle(m_node->handle());
    }

    virtual ~ComputeGraphKernelNode() = default;
};
}  // namespace muda