#pragma once
#include <map>
#include <muda/compute_graph/compute_graph_node_type.h>
#include <muda/compute_graph/compute_graph_node_id.h>
#include <muda/compute_graph/compute_graph_var_usage.h>
#include <muda/compute_graph/compute_graph_var_id.h>
namespace muda
{
class ComputeGraphNodeBase
{
  public:
    NodeId               node_id() const { return m_node_id; }
    ComputeGraphNodeType type() const { return m_type; }
    std::string_view     name() const { return m_name; }
    const auto&          var_usages() const { return m_var_usages; }
    virtual ~ComputeGraphNodeBase() = default;

  protected:
    friend class ComputeGraph;
    friend class ComputeGraphVarBase;
    ComputeGraphNodeBase(ComputeGraph*                           graph,
                         std::string_view                        name,
                         NodeId                                  node_id,
                         ComputeGraphNodeType                    type,
                         std::map<VarId, ComputeGraphVarUsage>&& usages)
        : m_node_id(node_id)
        , m_graph(graph)
        , m_name(name)
        , m_type(type)
        , m_var_usages(std::move(usages))
    {
    }
    std::map<VarId, ComputeGraphVarUsage> m_var_usages;
    NodeId                                m_node_id;
    ComputeGraph*                         m_graph;
    std::string                           m_name;
    ComputeGraphNodeType                  m_type;
};


}  // namespace muda