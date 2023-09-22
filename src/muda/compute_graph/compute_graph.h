#pragma once
#include <map>
#include <functional>
#include <muda/graph/graph.h>
#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/compute_graph/compute_graph_phase.h>

#include <muda/compute_graph/compute_graph_node_type.h>
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
    class TempNodeInfo
    {
      public:
        std::map<VarId, ComputeGraphVarUsage> var_usage;
    };
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
    ComputeGraphVar<T>* create_var(std::string_view name, T&& init_value);

    template <typename T>
    ComputeGraphVar<T>* find_var(std::string_view name);

    ComputeGraph& add_node(const Closure& f)
    {
        return add_node(std::string{"node"} + std::to_string(m_closures.size()), f);
    }

    ComputeGraph& add_node(std::string_view name, const Closure& f)
    {
        if(!m_allow_node_adding)
            throw muda::logic_error(  //
                "This graph is built or updated, so you can't add new nodes any more.");
        m_closures.emplace_back(name, f);
        return *this;
    }

    void update()
    {
        m_allow_node_adding = false;
        check_vars_valid();
        _update();
    }

    void launch(cudaStream_t s = nullptr);

    ClosureId current_closure_id() const { return m_current_closure_id; };

    NodeId current_node_id() const { return m_current_node_id; };

    ComputeGraphPhase current_graph_phase() const
    {
        return m_current_graph_phase;
    }

    //~ComputeGraph();
  private:
    void build();
    void _update();
    void check_vars_valid();

  private:
    friend class muda::details::ComputeGraphAccessor;


    bool                                  m_need_update = false;
    ClosureId                             m_current_closure_id;
    NodeId                                m_current_node_id;
    ComputeGraphPhase m_current_graph_phase = ComputeGraphPhase::None;
    bool              m_allow_access_graph  = false;
    bool              m_allow_node_adding   = true;
    TempNodeInfo      m_temp_node_info;

    Graph        m_graph;
    S<GraphExec> m_graph_exec{nullptr};
};
}  // namespace muda

namespace muda::details
{
// to prevent user access some internal function
class ComputeGraphAccessor
{
    ComputeGraph& m_cg;
    template <typename T>
    using S = std::shared_ptr<T>;

  public:
    ComputeGraphAccessor()
        : m_cg(*ComputeGraphBuilder::current_graph())
    {
    }

    ComputeGraphAccessor(ComputeGraph& graph)
        : m_cg(graph)
    {
    }
    ComputeGraphAccessor(ComputeGraph* graph)
        : m_cg(*graph)
    {
    }

    const auto& current_closure() const
    {
        return m_cg.m_closures[m_cg.current_closure_id().value()];
    }

    auto current_node() { return m_cg.m_nodes[m_cg.current_node_id().value()]; }

    const auto current_node() const
    {
        return m_cg.m_nodes[m_cg.current_node_id().value()];
    }

    template <typename T>
    void add_kernel_node(const S<KernelNodeParms<T>>& kernelParms);

    template <typename T>
    void update_kernel_node(const S<KernelNodeParms<T>>& kernelParms);

    void set_var_usage(VarId id, ComputeGraphVarUsage usage);

  private:
    template <typename F>
    void access_graph(F&& f)
    {
        if(!m_cg.m_allow_access_graph)
            throw std::runtime_error(  //
                "a graph closure can only contain one graph node");
        f(m_cg.m_graph);
        m_cg.m_allow_access_graph = false;
    }

    template <typename F>
    void access_graph_exec(F&& f)
    {
        if(!m_cg.m_allow_access_graph)
            throw std::runtime_error(  //
                "a graph closure can only contain one graph node");
        f(*m_cg.m_graph_exec.get());
        m_cg.m_allow_access_graph = false;
    }

    auto&& temp_var_usage() { return std::move(m_cg.m_temp_node_info.var_usage); }
};
}  // namespace muda::details

#include <muda/compute_graph/details/compute_graph.inl>