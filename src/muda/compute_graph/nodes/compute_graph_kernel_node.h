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
                           std::map<VarId, ComputeGraphVarUsage>&& usages,
                           S<KernelNode>                           m_node)
        : ComputeGraphNodeBase(graph, name, node_id, ComputeGraphNodeType::KernelNode, std::move(usages))
        , m_node(m_node)
    {
    }

    S<KernelNode> m_node;
};
}  // namespace muda