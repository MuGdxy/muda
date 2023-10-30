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
    ComputeGraphCaptureNode(NodeId node_id, uint64_t access_index)
        : ComputeGraphNodeBase("CaptureNode", node_id, access_index, ComputeGraphNodeType::CaptureNode)
    {
    }

    void set_node(cudaGraphNode_t node) { set_handle(node); }

    virtual ~ComputeGraphCaptureNode() = default;
};
}  // namespace muda