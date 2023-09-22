#pragma once
#include <map>
#include <muda/compute_graph/compute_graph_node_id.h>
#include <muda/compute_graph/compute_graph_var_usage.h>
#include <muda/compute_graph/compute_graph_var_id.h>
namespace muda
{
class ComputeGraphNodeBase
{
    std::map<VarId, ComputeGraphVarUsage> m_var_usages;
    NodeId                                m_node_id;
    ComputeGraph*                         m_graph;
    std::string                           m_name;

  public:
    NodeId node_id() const { return m_node_id; }

  protected:
    friend class ComputeGraph;
    ComputeGraphNodeBase(ComputeGraph* graph, std::string_view name, NodeId node_id)
        : m_node_id(node_id)
        , m_graph(graph)
        , m_name(name)
    {
    }
};

template <typename T>
class ComputeGraphNode : public ComputeGraphNodeBase
{
  protected:
    friend class ComputeGraph;
    using ComputeGraphNodeBase::ComputeGraphNodeBase;
};
}  // namespace muda