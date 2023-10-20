#pragma once
#include <muda/compute_graph/compute_graph_node.h>
#include <muda/graph/graph.h>

namespace muda
{
class ComputeGraphCaptureNode : public ComputeGraphNodeBase
{
  protected:
    friend class ComputeGraph;
    friend class details::ComputeGraphAccessor;
    ComputeGraphCaptureNode(ComputeGraph*                           graph,
                            std::string_view                        name,
                            NodeId                                  node_id,
                            std::map<VarId, ComputeGraphVarUsage>&& usages)
        : ComputeGraphNodeBase(
            graph, name, node_id, ComputeGraphNodeType::CaptureNode, std::move(usages))
    {
    }

    void set_node(cudaGraphNode_t node) { set_handle(node); }

    virtual ~ComputeGraphCaptureNode() = default;
};
}  // namespace muda