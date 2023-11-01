#pragma once
#include <map>
#include <functional>
#include <set>
#include <muda/mstl/span.h>
#include <muda/launch/event.h>
#include <muda/graph/graph.h>
#include <muda/launch/stream.h>
#include <muda/compute_graph/compute_graph_phase.h>
#include <muda/compute_graph/compute_graph_node_type.h>
#include <muda/compute_graph/compute_graph_node_id.h>
#include <muda/compute_graph/compute_graph_closure_id.h>
#include <muda/compute_graph/compute_graph_var_id.h>
#include <muda/compute_graph/compute_graph_var_usage.h>
#include <muda/compute_graph/compute_graph_dependency.h>
#include <muda/compute_graph/graphviz_options.h>
#include <muda/graph/graph_viewer.h>
namespace muda::details
{
class ComputeGraphAccessor;
}

namespace muda
{
class ComputeGraphVarBase;
class ComputeGraphNodeBase;
class ComputeGraphVarManager;
class ComputeGraphClosure;
template <typename T>
class ComputeGraphVar;

namespace details
{
    class LocalVarId : public U64IdWithType
    {
        using U64IdWithType::U64IdWithType;
    };
    class LocalVarInfo
    {
      public:
        LocalVarId           id{};
        ComputeGraphVarBase* var = nullptr;
    };
}  // namespace details

class ComputeGraph
{
  public:
    class AddNodeProxy
    {
        ComputeGraph& m_cg;
        std::string   m_node_name;

      public:
        AddNodeProxy(ComputeGraph& cg, std::string_view node_name);
        ComputeGraph& operator<<(std::function<void()>&& f) &&;
    };
    // A depends on B : from B to A
    using Dependency = ComputeGraphDependency;

    class GraphPhaseGuard
    {
        ComputeGraph& m_cg;

      public:
        GraphPhaseGuard(ComputeGraph& cg, ComputeGraphPhase phase);
        ~GraphPhaseGuard();
    };

    // delete copy
    ComputeGraph(const ComputeGraph&)            = delete;
    ComputeGraph& operator=(const ComputeGraph&) = delete;

    // move
    ComputeGraph(ComputeGraph&&)            = default;
    ComputeGraph& operator=(ComputeGraph&&) = default;

  private:
    //class TempNodeInfo
    //{
    //  public:
    //    std::map<VarId, ComputeGraphVarUsage> var_usage;
    //};
    template <typename T>
    using U = std::unique_ptr<T>;
    template <typename T>
    using S = std::shared_ptr<T>;

    friend class ComputeGraphVarBase;

    Graph        m_graph;
    S<GraphExec> m_graph_exec{nullptr};

    std::unordered_map<NodeId::value_type, cudaGraph_t> m_sub_graphs;

    std::vector<std::pair<std::string, ComputeGraphClosure*>> m_closures;

    std::map<VarId, details::LocalVarId> m_global_to_local_var_id;
    std::vector<details::LocalVarInfo>   m_related_vars;
    void emplace_related_var(ComputeGraphVarBase* var);


    std::vector<ComputeGraphNodeBase*>              m_nodes;
    std::vector<std::vector<ComputeGraphNodeBase*>> m_graph_nodes;
    std::vector<Dependency>                         m_deps;

    std::vector<int>        m_closure_need_update;
    ComputeGraphVarManager* m_var_manager = nullptr;

    friend class ComputeGraphVarManager;

    Event                      m_event;
    mutable Event::QueryResult m_event_result = Event::QueryResult::eFinished;

  public:
    ComputeGraph(ComputeGraphVarManager& manager, std::string_view name = "graph");

    ~ComputeGraph();

    /**************************************************************
    * 
    * Info API
    * 
    ***************************************************************/

    std::string_view name() const { return m_name; }

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

    Empty launch(bool single_stream, cudaStream_t s = nullptr);

    Empty launch(cudaStream_t s = nullptr) { return launch(false, s); }

    /**************************************************************
    * 
    * Graph Event Query API
    * 
    ***************************************************************/

    Event::QueryResult query() const;

    /**************************************************************
    * 
    * Graph Closure Capture Node API
    * 
    ***************************************************************/

    void capture(std::function<void(cudaStream_t)>&& f);

    /**************************************************************
    * 
    * Graph Visualization API
    * 
    ***************************************************************/

    void graphviz(std::ostream& o, const ComputeGraphGraphvizOptions& options = {});

    /**************************************************************
    * 
    * Graph Viewer API
    * 
    ***************************************************************/

    GraphViewer viewer();

    operator GraphViewer() { return viewer(); }

  private:  // internal method
    void topo_build();

    void build();

    void cuda_graph_add_deps();

    void build_deps();

    void serial_launch();

    void _update();

    void check_vars_valid();

    friend class AddNodeProxy;
    ComputeGraph& add_node(std::string&& name, const std::function<void()>& f);

    friend class ComputeGraphNodeBase;
    friend class ComputeGraphClosure;
    span<const Dependency> dep_span(size_t begin, size_t count) const;

    void set_current_graph_as_this();

    static void clear_current_graph();

    static Stream& shared_capture_stream();

    void add_capture_node(cudaGraph_t sub_graph);

    void update_capture_node(cudaGraph_t sub_graph);

    friend class ComputeGraphBuilder;
    ClosureId current_closure_id() const { return m_current_closure_id; };

    NodeId current_node_id() const { return m_current_node_id; };

    size_t current_access_index() const { return m_access_graph_index; }

    ComputeGraphPhase current_graph_phase() const;

  private:  // internal data
    friend class muda::details::ComputeGraphAccessor;
    std::string       m_name;
    bool              m_need_update = false;
    ClosureId         m_current_closure_id;
    NodeId            m_current_node_id;
    ComputeGraphPhase m_current_graph_phase = ComputeGraphPhase::None;
    bool              m_allow_access_graph  = false;
    size_t            m_access_graph_index  = 0;
    bool              m_allow_node_adding   = true;
    // TempNodeInfo      m_temp_node_info;
    cudaStream_t m_current_single_stream = nullptr;
    bool         m_is_capturing          = false;
    // in capture func, we don't allow any var eval()
    bool m_is_in_capture_func = false;
    // if we have already built the topo, we don't do that again
    bool m_is_topo_built = false;
};
}  // namespace muda


namespace muda
{
namespace details
{
    // allow devlopers to access some internal function
    class ComputeGraphAccessor
    {
        friend class ComputeGraph;
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

        auto& current_closure()
        {
            return m_cg.m_closures[m_cg.m_current_closure_id.value()];
        }

        //auto current_node()
        //{
        //    return m_cg.m_nodes[m_cg.current_node_id().value()];
        //}


        //const auto current_node() const
        //{
        //    return m_cg.m_nodes[m_cg.current_node_id().value()];
        //}

        const ComputeGraphNodeBase* current_node() const;


        ComputeGraphNodeBase* current_node();

        template <typename T>
        auto current_node()
        {
            return dynamic_cast<T*>(current_node());
        }

        /// <summary>
        /// automatically add or update kernel node by kernelParms (distincted by ComputeGraphPhase)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="kernelParms"></param>
        template <typename T>
        void set_kernel_node(const S<KernelNodeParms<T>>& kernelParms);

        void set_memcpy_node(void* dst, const void* src, size_t size_bytes, cudaMemcpyKind kind);

        void set_event_record_node(cudaEvent_t event);

        void set_event_wait_node(cudaEvent_t event);

        cudaStream_t current_stream() const
        {
            return m_cg.m_current_single_stream;
        }

        void check_allow_var_eval() const;

        void check_allow_node_adding() const;

        bool is_topo_built() const { return m_cg.m_is_topo_built; }

      private:
        friend class ComputeGraphVarBase;
        void set_var_usage(VarId id, ComputeGraphVarUsage usage);

        template <typename T>
        void add_kernel_node(const S<KernelNodeParms<T>>& kernelParms);
        template <typename T>
        void update_kernel_node(const S<KernelNodeParms<T>>& kernelParms);

        void add_memcpy_node(void* dst, const void* src, size_t size_bytes, cudaMemcpyKind kind);
        void update_memcpy_node(void* dst, const void* src, size_t size_bytes, cudaMemcpyKind kind);

        void add_event_record_node(cudaEvent_t event);
        void update_event_record_node(cudaEvent_t event);

        void add_event_wait_node(cudaEvent_t event);
        void update_event_wait_node(cudaEvent_t event);

        template <typename F>
        void access_graph(F&& f)
        {
            //if(!m_cg.m_allow_access_graph)
            //throw std::runtime_error(  //
            //    "a graph closure can only contain one graph node");
            f(m_cg.m_graph);
            //m_cg.m_allow_access_graph = false;
            ++m_cg.m_access_graph_index;
        }

        template <typename F>
        void access_graph_exec(F&& f)
        {
            //if(!m_cg.m_allow_access_graph)
            //    throw std::runtime_error(  //
            //        "a graph closure can only contain one graph node");
            f(*m_cg.m_graph_exec.get());
            //m_cg.m_allow_access_graph = false;
        }

        //auto&& temp_var_usage()
        //{
        //    return std::move(m_cg.m_temp_node_info.var_usage);
        //}

        template <typename NodeType, typename F>
        NodeType* get_or_create_node(F&& f);
    };
}  // namespace details
}  // namespace muda

#include "details/compute_graph.inl"