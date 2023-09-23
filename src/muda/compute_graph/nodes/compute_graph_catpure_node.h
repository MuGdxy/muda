#pragma once
#include <muda/compute_graph/compute_graph_node.h>
#include <muda/graph/graph.h>

namespace muda
{
class ComputeGraphCaptureNode : public ComputeGraphNodeBase
{
    template <typename T>
    using S = std::shared_ptr<T>;

  protected:
    friend class ComputeGraph;
    friend class details::ComputeGraphAccessor;
    ComputeGraphCaptureNode(ComputeGraph*                           graph,
                            std::string_view                        name,
                            NodeId                                  node_id,
                            std::map<VarId, ComputeGraphVarUsage>&& usages,
                            cudaGraphNode_t                         node)
        : ComputeGraphNodeBase(
            graph, name, node_id, ComputeGraphNodeType::CaptureNode, std::move(usages), node)
    {
    }
};
}  // namespace muda