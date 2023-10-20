#pragma once
#include <muda/compute_graph/compute_graph.h>
#include <muda/compute_graph/compute_graph_node.h>
#include <muda/graph/graph.h>
#include <muda/graph/memory_node.h>

namespace muda
{
//class ComputeGraphMemcpyNode : public ComputeGraphNodeBase
//{
//    template <typename T>
//    using S = std::shared_ptr<T>;
//
//  protected:
//    friend class ComputeGraph;
//    friend class details::ComputeGraphAccessor;
//    ComputeGraphMemcpyNode(ComputeGraph*                           graph,
//                           std::string_view                        name,
//                           NodeId                                  node_id,
//                           std::map<VarId, ComputeGraphVarUsage>&& usages)
//        : ComputeGraphNodeBase(
//            graph, name, node_id, ComputeGraphNodeType::MemcpyNode, std::move(usages))
//    {
//    }
//
//    S<MemcpyNode> m_node;
//    void          set_node(S<MemcpyNode> node)
//    {
//        m_node = node;
//        set_handle(m_node->handle());
//    }
//
//    virtual ~ComputeGraphMemcpyNode() = default;
//};

using ComputeGraphMemcpyNode =
    ComputeGraphNode<MemcpyNode, ComputeGraphNodeType::MemcpyNode>;
}  // namespace muda