#pragma once
#include <map>
#include <functional>
#include <muda/graph/graph.h>
#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/compute_graph/compute_graph_phase.h>

#include <muda/compute_graph/compute_graph_node_id.h>
#include <muda/compute_graph/compute_graph_closure_id.h>
#include <muda/compute_graph/compute_graph_var_id.h>
#include <muda/compute_graph/compute_graph_var_usage.h>

namespace muda::details
{
class ComputeGraphAccessor;
}

namespace muda
{
class ComputeGraphVarBase;
class ComputeGraphNodeBase;

template <typename T>
class ComputeGraphVar;

template <typename T>
class ComputeGraphNode;

class ComputeGraph
{
    using Closure = std::function<void()>;
    template <typename T>
    using U = std::unique_ptr<T>;
    template <typename T>
    using S = std::shared_ptr<T>;

    friend class ComputeGraphVarBase;

    std::vector<std::pair<std::string, Closure>>          m_closures;
    std::unordered_map<std::string, ComputeGraphVarBase*> m_vars_map;
    std::vector<ComputeGraphVarBase*>                     m_vars;
    std::vector<ComputeGraphNodeBase*>                    m_nodes;
    std::vector<int>                                      m_closure_need_update;

  public:
    template <typename T>
    ComputeGraphVar<T>* create_var(std::string_view name);

    template <typename T>
    ComputeGraphVar<T>* get_var(std::string_view name);

    ComputeGraph& add_node(const Closure& f) { return add_node("", f); }

    ComputeGraph& add_node(std::string_view name, const Closure& f)
    {
        m_closures.emplace_back(name, f);
        return *this;
    }

    void build();

    void update();

    ClosureId current_closure_id() const { return m_current_closure_id; };

    NodeId current_node_id() const { return m_current_node_id; };

    ComputeGraphPhase current_graph_phase() const
    {
        return m_current_graph_phase;
    }

    //~ComputeGraph();

  private:
    friend class muda::details::ComputeGraphAccessor;

    bool                                  m_need_update = false;
    std::map<VarId, ComputeGraphVarUsage> m_current_eval_id;
    ClosureId                             m_current_closure_id;
    NodeId                                m_current_node_id;
    ComputeGraphPhase m_current_graph_phase = ComputeGraphPhase::None;
    Graph             m_graph;
    S<GraphExec>      m_graph_exec{nullptr};
    bool              m_allow_access_graph = false;
};
}  // namespace muda

namespace muda::details
{
// to prevent user access some internal function
class ComputeGraphAccessor
{
    ComputeGraph& m_compute_graph;
    template <typename T>
    using S = std::shared_ptr<T>;

  public:
    ComputeGraphAccessor(ComputeGraph& graph)
        : m_compute_graph(graph)
    {
    }


    template <typename T>
    S<KernelNode> add_kernel_node(const S<KernelNodeParms<T>>& kernelParms)
    {
        return m_compute_graph.m_graph.add_node(kernelParms);
    }

    template <typename F>
    void access_graph(F&& f)
    {
        if(!m_compute_graph.m_allow_access_graph)
            throw std::runtime_error(  //
                "a graph closure can only contain one graph node");
        f(m_compute_graph.m_graph);
        m_compute_graph.m_allow_access_graph = false;
    }
};
}  // namespace muda::details

#include <muda/compute_graph/details/compute_graph.inl>