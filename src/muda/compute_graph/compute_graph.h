#pragma once
#include <map>
#include <functional>
#include <muda/graph/graph.h>
#include <muda/launch/stream.h>
#include <muda/compute_graph/compute_graph_phase.h>

#include <muda/compute_graph/compute_graph_node_type.h>
#include <muda/compute_graph/compute_graph_node_id.h>
#include <muda/compute_graph/compute_graph_closure_id.h>
#include <muda/compute_graph/compute_graph_var_id.h>
#include <muda/compute_graph/compute_graph_var_usage.h>

#include <muda/compute_graph/graphviz.h>

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

class ComputeGraphGraphvizOptions
{
  public:
    bool show_vars  = true;
    bool show_nodes = true;
};

class ComputeGraph
{
  public:
    using Closure = std::function<void()>;
    class AddNodeProxy
    {
        ComputeGraph& m_cg;
        std::string   m_node_name;

      public:
        AddNodeProxy(ComputeGraph& cg, std::string_view node_name);
        ComputeGraph& operator<<(ComputeGraph::Closure&& f) &&;
    };
    class Dependency
    {
      public:
        NodeId src;
        NodeId dst;
    };

    class DenpencySpan
    {
        const ComputeGraph& m_cg;
        size_t              m_begin;
        size_t              m_count;

      public:
        DenpencySpan(const ComputeGraph& cg, size_t begin, size_t count);

        const auto& operator[](size_t i) const;

        size_t count() const { return m_count; }
        size_t begin() const { return m_begin; }
    };

    class GraphPhaseGuard
    {
        ComputeGraph& m_cg;

      public:
        GraphPhaseGuard(ComputeGraph& cg, ComputeGraphPhase phase);
        ~GraphPhaseGuard();
    };

  private:
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

    Graph        m_graph;
    S<GraphExec> m_graph_exec{nullptr};

    std::unordered_map<NodeId::type, cudaGraph_t> m_sub_graphs;

    std::vector<std::pair<std::string, Closure>>          m_closures;
    std::unordered_map<std::string, ComputeGraphVarBase*> m_vars_map;
    std::vector<ComputeGraphVarBase*>                     m_vars;

    std::vector<ComputeGraphNodeBase*> m_nodes;
    std::vector<Dependency>            m_deps;

    std::vector<int> m_closure_need_update;

  public:
    /**************************************************************
    * 
    * GraphVar API
    * 
    ***************************************************************/
    template <typename T>
    ComputeGraphVar<T>& create_var(std::string_view name);

    template <typename T>
    ComputeGraphVar<T>& create_var(std::string_view name, T init_value);

    template <typename T>
    ComputeGraphVar<T>* find_var(std::string_view name);

    /**************************************************************
    * 
    * GraphNode API
    * 
    ***************************************************************/

    AddNodeProxy create_node(std::string_view node_name);


    /**************************************************************
    * 
    * Graph Launch API
    * 
    ***************************************************************/

    void update();

    void launch(bool single_stream, cudaStream_t s = nullptr);

    void launch(cudaStream_t s = nullptr) { launch(false, s); }

    /**************************************************************
    * 
    * Graph Closure Capture Node API
    * 
    ***************************************************************/

    cudaStream_t capture_stream();

    /**************************************************************
    * 
    * Graph Visualization API
    * 
    ***************************************************************/

    void graphviz(std::ostream& o, const ComputeGraphGraphvizOptions& options = {});

  private:  // internal method
    void build();

    void build_deps();

    void serial_launch();

    void _update();

    void check_vars_valid();

    friend class AddNodeProxy;
    ComputeGraph& add_node(std::string&& name, const Closure& f);

    friend class ComputeGraphNodeBase;
    DenpencySpan dep_span(size_t begin, size_t count) const;

    void set_current_graph_as_this();

    static void clear_current_graph();

    static Stream& shared_capture_stream();

    void add_capture_node(cudaGraph_t sub_graph);

    void update_capture_node(cudaGraph_t sub_graph);

    friend class ComputeGraphBuilder;
    ClosureId current_closure_id() const { return m_current_closure_id; };

    NodeId current_node_id() const { return m_current_node_id; };

    ComputeGraphPhase current_graph_phase() const;

  private:  // internal data
    friend class muda::details::ComputeGraphAccessor;

    bool              m_need_update = false;
    ClosureId         m_current_closure_id;
    NodeId            m_current_node_id;
    ComputeGraphPhase m_current_graph_phase = ComputeGraphPhase::None;
    bool              m_allow_access_graph  = false;
    bool              m_allow_node_adding   = true;
    TempNodeInfo      m_temp_node_info;
    cudaStream_t      m_current_single_stream = nullptr;
    bool              m_is_capturing          = false;
};
}  // namespace muda


namespace muda::details
{
// allow devlopers to access some internal function
class ComputeGraphAccessor
{
    ComputeGraph& m_cg;
    template <typename T>
    using S = std::shared_ptr<T>;

  public:
    ComputeGraphAccessor();

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

    /// <summary>
    /// automatically add or update kernel node by kernelParms (distincted by ComputeGraphPhase)
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="kernelParms"></param>
    template <typename T>
    void set_kernel_node(const S<KernelNodeParms<T>>& kernelParms);

    cudaStream_t current_stream() const { return m_cg.m_current_single_stream; }

  private:
    friend class ComputeGraphVarBase;
    void set_var_usage(VarId id, ComputeGraphVarUsage usage);

    template <typename T>
    void add_kernel_node(const S<KernelNodeParms<T>>& kernelParms);

    template <typename T>
    void update_kernel_node(const S<KernelNodeParms<T>>& kernelParms);

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

    auto&& temp_var_usage()
    {
        return std::move(m_cg.m_temp_node_info.var_usage);
    }
};
}  // namespace muda::details

#include <muda/compute_graph/details/compute_graph.inl>