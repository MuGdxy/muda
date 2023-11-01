#include <memory>
#include <muda/exception.h>
#include <muda/debug.h>
#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/compute_graph/compute_graph_var.h>
#include <muda/compute_graph/compute_graph_var_manager.h>
#include <muda/compute_graph/compute_graph_node.h>
#include <muda/compute_graph/nodes/compute_graph_kernel_node.h>
#include <muda/compute_graph/nodes/compute_graph_catpure_node.h>
#include <muda/compute_graph/nodes/compute_graph_memory_node.h>
#include <muda/compute_graph/nodes/compute_graph_event_node.h>
#include <muda/compute_graph/compute_graph_closure.h>

namespace muda
{
MUDA_INLINE ComputeGraph::AddNodeProxy::AddNodeProxy(ComputeGraph& cg, std::string_view node_name)
    : m_cg(cg)
    , m_node_name(node_name)
{
}

MUDA_INLINE ComputeGraph::GraphPhaseGuard::GraphPhaseGuard(ComputeGraph& cg, ComputeGraphPhase phase)
    : m_cg(cg)
{
    m_cg.set_current_graph_as_this();
    m_cg.m_current_graph_phase = phase;
}

MUDA_INLINE ComputeGraph::GraphPhaseGuard::~GraphPhaseGuard()
{
    m_cg.m_current_graph_phase = ComputeGraphPhase::None;
    ComputeGraph::clear_current_graph();
}

MUDA_INLINE ComputeGraph& ComputeGraph::AddNodeProxy::operator<<(std::function<void()>&& f) &&
{
    m_cg.add_node(std::move(m_node_name), f);
    return m_cg;
}

MUDA_INLINE void ComputeGraph::capture(std::function<void(cudaStream_t)>&& f)
{
    m_is_in_capture_func = true;
    switch(current_graph_phase())
    {
        case ComputeGraphPhase::TopoBuilding:
            // if this is called in topo building phase, we do nothing
            break;
        case ComputeGraphPhase::SerialLaunching: {
            // simply call it
            f(m_current_single_stream);
        }
        break;
        case ComputeGraphPhase::Updating:
        case ComputeGraphPhase::Building: {
            auto& s = shared_capture_stream();
            // begin capture and pass the stream to f
            s.begin_capture();
            m_is_capturing = true;
            f(s);
        }
        break;
        default:
            MUDA_ERROR_WITH_LOCATION("invoking capture() outside Graph Closure is not allowed");
            break;
    }
    m_is_in_capture_func = false;
}

MUDA_INLINE void ComputeGraph::graphviz(std::ostream& o, const ComputeGraphGraphvizOptions& options)
{
    topo_build();

    if(options.as_subgraph)
    {
        o << "subgraph cluster_" << options.graph_id;
        o << " {\n";
        o << "label=\"" << name() << "\";\n";
        o << options.cluster_style << "\n";
    }
    else
    {
        o << "digraph G {\n";
        o << options.graph_font << "\n";
    }


    if(options.show_vars && !options.as_subgraph)
    {
        o << "// vars: \n";
        for(auto&& [local_id, var] : m_related_vars)
        {
            var->graphviz_def(o, options);
            o << "\n";
        }
        o << "\n";
    }

    if(options.show_nodes)
    {
        o << "// nodes: \n";
        o << "node_g" << options.graph_id << "[label=\"" << name() << "\""
          << options.node_style << "]\n";

        for(auto& [name, node] : m_closures)
        {
            node->graphviz_def(o, options);
            o << "\n";
        }
        o << "\n";
        if(options.show_vars)
        {
            o << "// node var usages: \n";
            for(auto& [name, node] : m_closures)
            {
                node->graphviz_var_usages(o, options);
                o << "\n";
            }
            o << "\n";
        }
        o << "// node deps: \n";
        for(auto& [name, node] : m_closures)
        {
            if(node->deps().size() != 0)
                continue;
            o << "node_g" << options.graph_id << "->";
            node->graphviz_id(o, options);

            o << "[" << options.arc_style
              << "]"
                 "\n";
        }

        for(auto dep : m_deps)
        {
            auto src = m_closures[dep.to.value()];
            auto dst = m_closures[dep.from.value()];
            dst.second->graphviz_id(o, options);
            o << "->";
            src.second->graphviz_id(o, options);
            o << "[" << options.arc_style
              << "]"
                 "\n";
        }
    }
    o << "}\n";
}

MUDA_INLINE GraphViewer ComputeGraph::viewer()
{
    build();
    return GraphViewer{m_graph_exec->handle()};
}

MUDA_INLINE void ComputeGraph::topo_build()
{
    if(m_is_topo_built)
        return;

    m_closure_need_update.clear();
    m_closure_need_update.resize(m_closures.size(), false);
    GraphPhaseGuard guard(*this, ComputeGraphPhase::TopoBuilding);
    for(size_t i = 0; i < m_closures.size(); ++i)
    {
        //m_current_node_id    = NodeId{i};
        m_current_closure_id = ClosureId{i};
        m_allow_access_graph = true;
        m_access_graph_index = 0;
        m_closures[i].second->operator()();
        if(m_is_capturing)
            add_capture_node(m_sub_graphs[i]);
    }

    build_deps();
}

MUDA_INLINE void ComputeGraph::build()
{
    if(m_graph_exec)
        return;

    GraphPhaseGuard guard(*this, ComputeGraphPhase::Building);
    if(!m_is_topo_built)
    {
        m_closure_need_update.clear();
        m_closure_need_update.resize(m_closures.size(), false);
    }
    for(size_t i = 0; i < m_closures.size(); ++i)
    {
        //m_current_node_id    = NodeId{i};
        m_current_closure_id = ClosureId{i};
        m_allow_access_graph = true;
        m_access_graph_index = 0;
        m_closures[i].second->operator()();
        if(m_is_capturing)
            add_capture_node(m_sub_graphs[i]);
    }
    if(!m_is_topo_built)
        build_deps();
    cuda_graph_add_deps();

    m_graph_exec = m_graph.instantiate();
}

MUDA_INLINE void ComputeGraph::serial_launch()
{
    GraphPhaseGuard guard(*this, ComputeGraphPhase::SerialLaunching);

    for(size_t i = 0; i < m_closures.size(); ++i)
    {
        // m_current_node_id    = NodeId{i};
        m_current_closure_id = ClosureId{i};
        m_allow_access_graph = false;  // no need to access graph
        m_closures[i].second->operator()();
        m_is_capturing = false;
    }
}

MUDA_INLINE void ComputeGraph::check_vars_valid()
{
    for(auto&& [local_id, var] : m_related_vars)
        if(!var->is_valid())
        {
            MUDA_ERROR_WITH_LOCATION(
                "var[%s] is not valid, "
                "you need update the var before launch this graph",
                var->name().data());
        }
}

MUDA_INLINE ComputeGraph& ComputeGraph::add_node(std::string&& name,
                                                 const std::function<void()>& f)
{
    details::ComputeGraphAccessor(this).check_allow_node_adding();
    if(!m_allow_node_adding)
        MUDA_ERROR_WITH_LOCATION("This graph is built or updated, so you can't add new nodes any more.");
    auto size    = m_closures.size();
    auto closure = new ComputeGraphClosure{this, ClosureId{size}, name, f};
    m_closures.emplace_back(name, closure);
    return *this;
}

MUDA_INLINE auto ComputeGraph::dep_span(size_t begin, size_t count) const
    -> span<const Dependency>
{
    return span<const Dependency>{m_deps}.subspan(begin, count);
}

MUDA_INLINE void ComputeGraph::set_current_graph_as_this()
{
    ComputeGraphBuilder::current_graph(this);
}

MUDA_INLINE void ComputeGraph::clear_current_graph()
{
    ComputeGraphBuilder::current_graph(nullptr);
}

MUDA_INLINE Stream& ComputeGraph::shared_capture_stream()
{
    static thread_local Stream s(Stream::Flag::eNonBlocking);
    return s;
}

MUDA_INLINE void ComputeGraph::add_capture_node(cudaGraph_t sub_graph)
{
    auto capture_node = details::ComputeGraphAccessor(this).get_or_create_node<ComputeGraphCaptureNode>(
        [&]
        {
            return new ComputeGraphCaptureNode{NodeId{m_nodes.size()}, m_access_graph_index};
        });
    if(ComputeGraphBuilder::is_building())
    {

        shared_capture_stream().end_capture(&sub_graph);
        cudaGraphNode_t node;
        checkCudaErrors(cudaGraphAddChildGraphNode(&node, m_graph.handle(), nullptr, 0, sub_graph));
        capture_node->set_node(node);
    }
    m_is_capturing = false;
}

MUDA_INLINE void ComputeGraph::update_capture_node(cudaGraph_t sub_graph)
{
    const auto& [name, closure] = m_closures[current_closure_id().value()];
    auto node                   = m_nodes[current_node_id().value()];
    auto capture_node           = dynamic_cast<ComputeGraphCaptureNode*>(node);
    auto cuda_node              = capture_node->handle();
    checkCudaErrors(cudaGraphExecChildGraphNodeSetParams(m_graph_exec->handle(), cuda_node, sub_graph));
    m_is_capturing = false;
}

MUDA_INLINE ComputeGraphPhase ComputeGraph::current_graph_phase() const
{
    return m_current_graph_phase;
}

MUDA_INLINE void ComputeGraph::_update()
{
    if(!m_need_update)
        return;

    GraphPhaseGuard guard(*this, ComputeGraphPhase::Updating);

    for(size_t i = 0; i < m_closure_need_update.size(); ++i)
    {
        auto& need_update = m_closure_need_update[i];
        if(need_update)
        {
            m_current_closure_id = ClosureId{i};
            // m_current_node_id    = NodeId{i};
            m_allow_access_graph = true;
            m_access_graph_index = 0;
            m_closures[i].second->operator()();
            if(m_is_capturing)
                update_capture_node(m_sub_graphs[i]);
            m_is_capturing = false;
            need_update    = false;
        }
    }
}

MUDA_INLINE ComputeGraph::~ComputeGraph()
{
    for(auto var_info : m_related_vars)
        var_info.var->remove_related_closure_infos(this);

    m_var_manager->m_graphs.erase(this);

    for(auto node : m_nodes)
        delete node;

    for(auto& [name, closure] : m_closures)
        delete closure;
}

MUDA_INLINE void ComputeGraph::emplace_related_var(ComputeGraphVarBase* var)
{
    auto global_var_id = var->var_id();
    auto iter          = m_global_to_local_var_id.find(global_var_id);
    if(iter == m_global_to_local_var_id.end())
    {
        auto local_id = details::LocalVarId{m_related_vars.size()};
        m_related_vars.emplace_back(details::LocalVarInfo{local_id, var});
        m_global_to_local_var_id.emplace(std::make_pair(global_var_id, local_id));
    }
}

MUDA_INLINE ComputeGraph::ComputeGraph(ComputeGraphVarManager& manager, std::string_view name)
    : m_var_manager(&manager)
    , m_name(name)
{
    m_var_manager->m_graphs.insert(this);
}


MUDA_INLINE ComputeGraph::AddNodeProxy ComputeGraph::create_node(std::string_view node_name)
{
    return AddNodeProxy{*this, node_name};
}

MUDA_INLINE void ComputeGraph::update()
{
    m_allow_node_adding = false;
    check_vars_valid();
    _update();
}

MUDA_INLINE Empty ComputeGraph::launch(bool single_stream, cudaStream_t s)
{
    m_allow_node_adding = false;
    if(single_stream)
    {
        m_current_single_stream = s;
        serial_launch();
    }
    else
    {
        check_vars_valid();
        build();
        _update();
        m_graph_exec->launch(s);
    }
    m_event_result = Event::QueryResult::eNotReady;
    on(s).record(m_event);
#if MUDA_CHECK_ON
    if(Debug::is_debug_sync_all())
        checkCudaErrors(cudaStreamSynchronize(s));
#endif
    return Empty{s};
}

MUDA_INLINE Event::QueryResult ComputeGraph::query() const
{
    if(m_event_result == Event::QueryResult::eNotReady)
        m_event_result = m_event.query();
    return m_event_result;
}

namespace details
{
    MUDA_INLINE ComputeGraphAccessor::ComputeGraphAccessor()
        : m_cg(*ComputeGraphBuilder::current_graph())
    {
    }

    MUDA_INLINE void ComputeGraphAccessor::check_allow_var_eval() const
    {
        if(m_cg.m_is_in_capture_func)
            MUDA_ERROR_WITH_LOCATION("you can't eval a var in ComputeGraph::capture() function");
    }

    MUDA_INLINE void ComputeGraphAccessor::check_allow_node_adding() const
    {
        if(m_cg.current_graph_phase() != ComputeGraphPhase::None)
            MUDA_ERROR_WITH_LOCATION("you are not allowed adding node at this point");
    }

    /*
     * Set Graph Node 
     */
    template <typename T>
    MUDA_INLINE void ComputeGraphAccessor::set_kernel_node(const S<KernelNodeParms<T>>& kernelParms)
    {
        switch(ComputeGraphBuilder::current_phase())
        {
            case ComputeGraphPhase::TopoBuilding:
                MUDA_ASSERT(!kernelParms,
                            "When ComputeGraphPhase == TopoBuilding, "
                            "you don't need to create NodeParms, so keep it nullptr.");
                // fall through
            case ComputeGraphPhase::Building:
                add_kernel_node(kernelParms);
                break;
            case ComputeGraphPhase::Updating:
                update_kernel_node(kernelParms);
                break;
            default:
                MUDA_ERROR_WITH_LOCATION("invalid phase");
                break;
        }
    }
    template <typename T>
    MUDA_INLINE void ComputeGraphAccessor::add_kernel_node(const S<KernelNodeParms<T>>& parms)
    {
        access_graph([&](Graph& g) {  // create kernel node
            ComputeGraphKernelNode* kernel_node = get_or_create_node<ComputeGraphKernelNode>(
                [&]
                {
                    const auto& [name, closure] = current_closure();
                    return new ComputeGraphKernelNode(NodeId{m_cg.m_nodes.size()},
                                                      m_cg.current_access_index());
                });
            if(ComputeGraphBuilder::current_phase() == ComputeGraphPhase::Building)
            {
                kernel_node->set_node(g.add_kernel_node(parms));
            }
        });
    }
    template <typename T>
    MUDA_INLINE void ComputeGraphAccessor::update_kernel_node(const S<KernelNodeParms<T>>& kernelParms)
    {
        access_graph_exec(
            [&](GraphExec& g_exec)
            {
                const auto& [name, closure] = current_closure();
                auto kernel_node = current_node<ComputeGraphKernelNode>();
                g_exec.set_kernel_node_parms(kernel_node->m_node, kernelParms);
            });
    }

    MUDA_INLINE const ComputeGraphNodeBase* ComputeGraphAccessor::current_node() const
    {
        return current_closure().second->m_graph_nodes[m_cg.current_access_index()];
    }

    MUDA_INLINE ComputeGraphNodeBase* ComputeGraphAccessor::current_node()
    {
        return current_closure().second->m_graph_nodes[m_cg.current_access_index()];
    }

    MUDA_INLINE void ComputeGraphAccessor::set_memcpy_node(void*       dst,
                                                           const void* src,
                                                           size_t size_bytes,
                                                           cudaMemcpyKind kind)
    {
        switch(ComputeGraphBuilder::current_phase())
        {
            case ComputeGraphPhase::TopoBuilding:
                // fall through
            case ComputeGraphPhase::Building:
                add_memcpy_node(dst, src, size_bytes, kind);
                break;
            case ComputeGraphPhase::Updating:
                update_memcpy_node(dst, src, size_bytes, kind);
                break;
            default:
                MUDA_ERROR_WITH_LOCATION("invalid phase");
                break;
        }
    }
    MUDA_INLINE void ComputeGraphAccessor::add_memcpy_node(void*       dst,
                                                           const void* src,
                                                           size_t size_bytes,
                                                           cudaMemcpyKind kind)
    {
        access_graph([&](Graph& g) {  // create memory node
            ComputeGraphMemcpyNode* memory_node = get_or_create_node<ComputeGraphMemcpyNode>(
                [&]
                {
                    const auto& [name, closure] = current_closure();
                    return new ComputeGraphMemcpyNode(NodeId{m_cg.m_nodes.size()},
                                                      m_cg.current_access_index());
                });
            if(ComputeGraphBuilder::current_phase() == ComputeGraphPhase::Building)
                memory_node->set_node(g.add_memcpy_node(dst, src, size_bytes, kind));
        });
    }
    MUDA_INLINE void ComputeGraphAccessor::update_memcpy_node(void*       dst,
                                                              const void* src,
                                                              size_t size_bytes,
                                                              cudaMemcpyKind kind)
    {
        access_graph_exec(
            [&](GraphExec& g_exec)
            {
                const auto& [name, closure] = current_closure();
                auto memory_node = current_node<ComputeGraphMemcpyNode>();
                g_exec.set_memcpy_node_parms(memory_node->m_node, dst, src, size_bytes, kind);
            });
    }


    MUDA_INLINE void ComputeGraphAccessor::set_event_record_node(cudaEvent_t event)
    {
        switch(ComputeGraphBuilder::current_phase())
        {
            case ComputeGraphPhase::TopoBuilding:
                MUDA_ASSERT(!event,
                            "When ComputeGraphPhase == TopoBuilding, "
                            "you don't need to create event, so keep it nullptr.");
                // fall through
            case ComputeGraphPhase::Building:
                add_event_record_node(event);
                break;
            case ComputeGraphPhase::Updating:
                update_event_record_node(event);
                break;
            default:
                MUDA_ERROR_WITH_LOCATION("invalid phase");
                break;
        }
    }
    MUDA_INLINE void ComputeGraphAccessor::add_event_record_node(cudaEvent_t event)
    {
        access_graph(
            [&](Graph& g)
            {
                ComputeGraphEventRecordNode* event_record =
                    get_or_create_node<ComputeGraphEventRecordNode>(
                        [&]
                        {
                            const auto& [name, closure] = current_closure();
                            return new ComputeGraphEventRecordNode(
                                NodeId{m_cg.m_nodes.size()}, m_cg.current_access_index());
                        });

                if(ComputeGraphBuilder::current_phase() == ComputeGraphPhase::Building)
                {
                    event_record->set_node(g.add_event_record_node(event));
                }
            });
    }
    MUDA_INLINE void ComputeGraphAccessor::update_event_record_node(cudaEvent_t event)
    {
        access_graph_exec(
            [&](GraphExec& g_exec)
            {
                const auto& [name, closure] = current_closure();
                auto event_record = current_node<ComputeGraphEventRecordNode>();
                g_exec.set_event_record_node_parms(event_record->m_node, event);
            });
    }


    MUDA_INLINE void ComputeGraphAccessor::set_event_wait_node(cudaEvent_t event)
    {
        switch(ComputeGraphBuilder::current_phase())
        {
            case ComputeGraphPhase::TopoBuilding:
                MUDA_ASSERT(!event,
                            "When ComputeGraphPhase == TopoBuilding, "
                            "you don't need to create event, so keep it nullptr.");
                // fall through
            case ComputeGraphPhase::Building:
                add_event_wait_node(event);
                break;
            case ComputeGraphPhase::Updating:
                update_event_wait_node(event);
                break;
            default:
                MUDA_ERROR_WITH_LOCATION("invalid phase");
                break;
        }
    }
    MUDA_INLINE void ComputeGraphAccessor::add_event_wait_node(cudaEvent_t event)
    {
        access_graph(
            [&](Graph& g)
            {
                ComputeGraphEventWaitNode* event_wait =
                    get_or_create_node<ComputeGraphEventWaitNode>(
                        [&]
                        {
                            const auto& [name, closure] = current_closure();
                            return new ComputeGraphEventWaitNode(
                                NodeId{m_cg.m_nodes.size()}, m_cg.current_access_index());
                        });

                if(ComputeGraphBuilder::current_phase() == ComputeGraphPhase::Building)
                {
                    event_wait->set_node(g.add_event_wait_node(event));
                }
            });
    }
    MUDA_INLINE void ComputeGraphAccessor::update_event_wait_node(cudaEvent_t event)
    {
        access_graph_exec(
            [&](GraphExec& g_exec)
            {
                const auto& [name, closure] = current_closure();
                auto event_wait = current_node<ComputeGraphEventWaitNode>();
                g_exec.set_event_wait_node_parms(event_wait->m_node, event);
            });
    }


    template <typename NodeType, typename F>
    MUDA_INLINE NodeType* ComputeGraphAccessor::get_or_create_node(F&& f)
    {
        static_assert(std::is_base_of_v<ComputeGraphNodeBase, NodeType>,
                      "NodeType must be derived from ComputeGraphNodeBase");
        if(!m_cg.m_is_topo_built)
        {
            NodeType* ptr         = f();
            auto& [name, closure] = current_closure();
            closure->m_graph_nodes.emplace_back(ptr);
            m_cg.m_nodes.emplace_back(ptr);
            return ptr;
        }
        else
            return current_node<NodeType>();
    }
    MUDA_INLINE void ComputeGraphAccessor::set_var_usage(VarId id, ComputeGraphVarUsage usage)
    {
        auto& dst_usage = current_closure().second->m_var_usages[id];
        if(dst_usage < usage)
            dst_usage = usage;
    }

}  // namespace details
}  // namespace muda

namespace muda
{
/*
 * Build Graph Dependencies
 */
namespace details
{
    MUDA_INLINE void process_node(std::vector<ComputeGraph::Dependency>& deps,
                                  std::vector<ClosureId>& last_read_or_write_nodes,
                                  std::vector<ClosureId>& last_write_nodes,
                                  ComputeGraphClosure&    closure,
                                  const std::vector<std::pair<LocalVarId, ComputeGraphVarUsage>>& local_var_usage,
                                  uint64_t& dep_begin,
                                  uint64_t& dep_count)
    {
        auto is_read_write = [](ComputeGraphVarUsage usage)
        { return usage == ComputeGraphVarUsage::ReadWrite; };
        auto is_read_only = [](ComputeGraphVarUsage usage)
        { return usage == ComputeGraphVarUsage::Read; };

        std::unordered_set<ClosureId> unique_deps;

        for(auto& [local_var_id, usage] : local_var_usage)
        {
            // if this is a written resource,
            // this should depend on any write and read before it
            // to get newest data or to avoid data corruption
            if(is_read_write(usage))
            {
                auto dst_nid = last_read_or_write_nodes[local_var_id.value()];
                if(dst_nid.is_valid())
                {
                    // the last accessing node reads or writes this resrouce, so I should depend on it
                    if(unique_deps.find(dst_nid) == unique_deps.end())
                    {
                        // record this dependency
                        unique_deps.insert(dst_nid);
                    }
                }
            }
            // if this is a read resource,
            // this should depend on any write before it
            // to get newest data
            // but it has no need to depend on any read before it
            else if(is_read_only(usage))
            {
                auto dst_nid = last_write_nodes[local_var_id.value()];
                if(dst_nid.is_valid())
                {
                    // the last accessing node writes this resrouce, so I should depend on it
                    if(unique_deps.find(dst_nid) == unique_deps.end())
                    {
                        // record this dependency
                        unique_deps.insert(dst_nid);
                    }
                }
            }
        }

        auto current_closure_id = closure.clousure_id();

        // set up res node map with pair [res, node]
        for(auto& [local_var_id, usage] : local_var_usage)
        {
            // if this is a write resource,
            // the latter read/write kernel should depend on this
            // to get the newest data.
            if(is_read_write(usage))
            {
                last_read_or_write_nodes[local_var_id.value()] = current_closure_id;
                last_write_nodes[local_var_id.value()] = current_closure_id;
            }
            // if this is a read resource,
            // the latter write kernel should depend on this
            // to avoid data corruption.
            else if(is_read_only(usage))
            {
                last_read_or_write_nodes[local_var_id.value()] = current_closure_id;
            }
        }

        // add dependencies to deps
        dep_begin = deps.size();
        for(auto dep : unique_deps)
            deps.emplace_back(ComputeGraph::Dependency{dep, current_closure_id});
        dep_count = unique_deps.size();
    }
}  // namespace details

MUDA_INLINE void ComputeGraph::cuda_graph_add_deps()
{
    std::vector<cudaGraphNode_t> froms;
    froms.reserve(m_deps.size());
    std::vector<cudaGraphNode_t> tos;
    tos.reserve(m_deps.size());

    // in closure deps

    for(auto& [name, closure] : m_closures)
    {
        MUDA_ASSERT(closure->m_graph_nodes.size() > 0, "closure[%s] has no nodes", name.data());
        if(closure->m_graph_nodes.size() == 1)
            continue;

        auto from = closure->m_graph_nodes.front();
        auto to   = from;
        for(size_t i = 1; i < closure->m_graph_nodes.size(); ++i)
        {
            to = closure->m_graph_nodes[i];
            froms.emplace_back(from->handle());
            tos.emplace_back(to->handle());
            from = to;
        }
    }


    for(auto dep : m_deps)
    {
        auto from = m_closures[dep.from.value()].second->m_graph_nodes.back();
        auto to   = m_closures[dep.to.value()].second->m_graph_nodes.front();


        froms.emplace_back(from->handle());
        tos.emplace_back(to->handle());
    };

    checkCudaErrors(cudaGraphAddDependencies(
        m_graph.handle(), froms.data(), tos.data(), froms.size()));
}

MUDA_INLINE void ComputeGraph::build_deps()
{
    m_deps.clear();
    auto local_var_count = m_related_vars.size();

    // map: var_id -> node_id, uint64_t{-1} means no write node yet
    auto last_write_nodes = std::vector<ClosureId>(local_var_count, ClosureId{});
    // map: var_id -> node_id, uint64_t{-1} means no read node yet
    auto last_read_or_write_nodes = std::vector<ClosureId>(local_var_count, ClosureId{});

    // process all nodes
    for(size_t i = 0u; i < m_closures.size(); i++)
    {
        auto& [name, closure] = m_closures[i];

        // map global var id to local var id
        std::vector<std::pair<details::LocalVarId, ComputeGraphVarUsage>> local_var_usage;
        local_var_usage.reserve(closure->var_usages().size());
        for(auto&& [var_id, usage] : closure->var_usages())
        {
            auto local_id = m_global_to_local_var_id[var_id];
            local_var_usage.emplace_back(local_id, usage);
        }

        size_t dep_begin, dep_count;
        details::process_node(
            m_deps, last_read_or_write_nodes, last_write_nodes, *closure, local_var_usage, dep_begin, dep_count);
        closure->set_deps_range(dep_begin, dep_count);
    }

    m_is_topo_built = true;
}
}  // namespace muda