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
    auto        node_id() const { return m_node_id; }
    auto        type() const { return m_type; }
    auto        name() const { return std::string_view{m_name}; }
    const auto& var_usages() const { return m_var_usages; }
    auto deps() const { return m_graph->dep_span(m_deps_begin, m_deps_count); }

    virtual void graphviz_id(std::ostream& o) const;
    virtual void graphviz_def(std::ostream& o) const;
    virtual void graphviz_var_usages(std::ostream& o) const;
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
    size_t                                m_deps_begin = 0;
    size_t                                m_deps_count = 0;
    cudaGraphNode_t                       m_cuda_node  = nullptr;

    void set_deps_range(size_t begin, size_t count);
    auto handle() const { return m_cuda_node; }
    void set_handle(cudaGraphNode_t handle) { m_cuda_node = handle; }
    auto is_valid() const { return m_cuda_node; }
};
}  // namespace muda

#include <muda/compute_graph/details/compute_graph_node.inl>