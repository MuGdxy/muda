#pragma once
#include <memory>

#include <muda/exception.h>
#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/compute_graph/compute_graph_var.h>
#include <muda/compute_graph/compute_graph_node.h>
#include <muda/compute_graph/nodes/compute_graph_kernel_node.h>
#include <muda/compute_graph/nodes/compute_graph_catpure_node.h>
#include <muda/compute_graph/nodes/compute_graph_memory_node.h>

namespace muda
{
MUDA_INLINE ComputeGraph::AddNodeProxy::AddNodeProxy(ComputeGraph& cg, std::string_view node_name)
    : m_cg(cg)
    , m_node_name(node_name)
{
}

MUDA_INLINE ComputeGraph::DenpencySpan::DenpencySpan(const ComputeGraph& cg, size_t begin, size_t count)
    : m_cg(cg)
    , m_begin(begin)
    , m_count(count)
{
}

MUDA_INLINE const auto& ComputeGraph::DenpencySpan::operator[](size_t i) const
{
    return m_cg.m_deps[m_begin + i];
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

MUDA_INLINE ComputeGraph& ComputeGraph::AddNodeProxy::operator<<(ComputeGraph::Closure&& f) &&
{
    m_cg.add_node(std::move(m_node_name), f);
    return m_cg;
}

template <typename T>
MUDA_INLINE ComputeGraphVar<T>& ComputeGraph::create_var(std::string_view name)
{
    auto ptr = new ComputeGraphVar<T>(this, name, VarId{m_vars.size()});
    m_vars.emplace_back(ptr);
    if(m_vars_map.find(std::string{name}) != m_vars_map.end())
        MUDA_ERROR_WITH_LOCATION("var[%s] already exists", name.data());
    m_vars_map.emplace(name, ptr);
    return *ptr;
}
template <typename T>
MUDA_INLINE ComputeGraphVar<T>& ComputeGraph::create_var(std::string_view name, T init_value)
{
    auto ptr = new ComputeGraphVar<T>(this, name, VarId{m_vars.size()}, init_value);
    m_vars.emplace_back(ptr);
    m_vars_map.emplace(name, ptr);
    return *ptr;
}
template <typename T>
MUDA_INLINE ComputeGraphVar<T>* ComputeGraph::find_var(std::string_view name)
{
    auto it = m_vars_map.find(std::string{name});
    if(it == m_vars_map.end())
        return nullptr;
    return dynamic_cast<ComputeGraphVar<T>*>(it->second);
};

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
    o << "digraph G {\n"
         "beautify=true";
    if(options.show_vars)
    {
        o << "// vars: \n";
        for(auto&& var : m_vars)
        {
            var->graphviz_def(o);
            o << "\n";
        }
        o << "\n";
    }

    if(options.show_nodes)
    {
        o << "// nodes: \n";
        for(auto&& node : m_nodes)
        {
            node->graphviz_def(o);
            o << "\n";
        }
        o << "\n";
        if(options.show_vars)
        {
            o << "// node var usages: \n";
            for(auto&& node : m_nodes)
            {
                node->graphviz_var_usages(o);
                o << "\n";
            }
            o << "\n";
        }
        o << "// node deps: \n";
        for(auto dep : m_deps)
        {
            auto src = m_nodes[dep.src.value()];
            auto dst = m_nodes[dep.dst.value()];
            dst->graphviz_id(o);
            o << "->";
            src->graphviz_id(o);
            o << R"([color="#82B366"])"
                 "\n";
        }
    }
    o << "}\n";
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
        m_current_node_id    = NodeId{i};
        m_current_closure_id = ClosureId{i};
        m_allow_access_graph = true;
        m_closures[i].second();
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
        m_current_node_id    = NodeId{i};
        m_current_closure_id = ClosureId{i};
        m_allow_access_graph = true;
        m_closures[i].second();
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
        m_current_node_id    = NodeId{i};
        m_current_closure_id = ClosureId{i};
        m_allow_access_graph = false;  // no need to access graph
        m_closures[i].second();
        m_is_capturing = false;
    }
}

MUDA_INLINE void ComputeGraph::check_vars_valid()
{
    for(auto& var : m_vars)
        if(!var->is_valid())
            MUDA_ERROR_WITH_LOCATION(
                "var[%s] is not valid, "
                "you need update the var before launch this graph",
                var->name().data());
}

MUDA_INLINE ComputeGraph& ComputeGraph::add_node(std::string&& name, const Closure& f)
{
    details::ComputeGraphAccessor(this).check_allow_node_adding();
    if(!m_allow_node_adding)
        MUDA_ERROR_WITH_LOCATION("This graph is built or updated, so you can't add new nodes any more.");
    m_closures.emplace_back(std::move(name), f);
    return *this;
}

MUDA_INLINE auto ComputeGraph::dep_span(size_t begin, size_t count) const -> DenpencySpan
{
    return DenpencySpan{*this, begin, count};
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
    static thread_local Stream s(Stream::flag::eNonBlocking);
    return s;
}

MUDA_INLINE void ComputeGraph::add_capture_node(cudaGraph_t sub_graph)
{
    auto capture_node = details::ComputeGraphAccessor(this).get_or_create_node<ComputeGraphCaptureNode>(
        [&]
        {
            const auto& [name, closure] = m_closures[current_closure_id().value()];
            return new ComputeGraphCaptureNode{this,
                                               name,
                                               NodeId{m_nodes.size()},  // node id
                                               std::move(m_temp_node_info.var_usage)};
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
            m_current_node_id    = NodeId{i};
            m_allow_access_graph = true;
            m_closures[i].second();
            if(m_is_capturing)
                update_capture_node(m_sub_graphs[i]);
            m_is_capturing = false;
            need_update    = false;
        }
    }
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

MUDA_INLINE void ComputeGraph::launch(bool single_stream, cudaStream_t s)
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
}

template <typename T>
MUDA_INLINE void details::ComputeGraphAccessor::add_kernel_node(const S<KernelNodeParms<T>>& parms)
{
    access_graph([&](Graph& g) {  // create kernel node
        ComputeGraphKernelNode* kernel_node = get_or_create_node<ComputeGraphKernelNode>(
            [&]
            {
                const auto& [name, closure] = current_closure();
                return new ComputeGraphKernelNode(  //
                    &m_cg,                          // compute graph
                    name,                           // node name
                    NodeId{m_cg.m_nodes.size()},    // node id
                    temp_var_usage());              // kernel node
            });
        if(ComputeGraphBuilder::current_phase() == ComputeGraphPhase::Building)
            kernel_node->set_node(g.add_kernel_node(parms));
    });
}

template <typename T>
MUDA_INLINE void details::ComputeGraphAccessor::update_kernel_node(const S<KernelNodeParms<T>>& kernelParms)
{
    access_graph_exec(
        [&](GraphExec& g_exec)
        {
            const auto& [name, closure] = current_closure();
            auto kernel_node = current_node<ComputeGraphKernelNode>();
            g_exec.set_kernel_node_parms(kernel_node->m_node, kernelParms);
        });
}

MUDA_INLINE void details::ComputeGraphAccessor::add_memcpy_node(void*       dst,
                                                                const void* src,
                                                                size_t size_bytes,
                                                                cudaMemcpyKind kind)
{
    access_graph([&](Graph& g) {  // create memory node
        ComputeGraphMemcpyNode* memory_node = get_or_create_node<ComputeGraphMemcpyNode>(
            [&]
            {
                const auto& [name, closure] = current_closure();
                return new ComputeGraphMemcpyNode(  //
                    &m_cg,                          // compute graph
                    name,                           // node name
                    NodeId{m_cg.m_nodes.size()},    // node id
                    temp_var_usage());
            });
        if(ComputeGraphBuilder::current_phase() == ComputeGraphPhase::Building)
            memory_node->set_node(g.add_memcpy_node(dst, src, size_bytes, kind));
    });
}

MUDA_INLINE void details::ComputeGraphAccessor::update_memcpy_node(
    void* dst, const void* src, size_t size_bytes, cudaMemcpyKind kind)
{
    access_graph_exec(
        [&](GraphExec& g_exec)
        {
            const auto& [name, closure] = current_closure();
            auto memory_node = current_node<ComputeGraphMemcpyNode>();
            g_exec.set_memcpy_node_parms(memory_node->m_node, dst, src, size_bytes, kind);
        });
}

template <typename NodeType, typename F>
MUDA_INLINE NodeType* details::ComputeGraphAccessor::get_or_create_node(F&& f)
{
    static_assert(std::is_base_of_v<ComputeGraphNodeBase, NodeType>,
                  "NodeType must be derived from ComputeGraphNodeBase");
    if(!m_cg.m_is_topo_built)
    {
        NodeType* ptr = f();
        m_cg.m_nodes.emplace_back(ptr);
        return ptr;
    }
    else
        return current_node<NodeType>();
}

template <typename T>
MUDA_INLINE void details::ComputeGraphAccessor::set_kernel_node(const S<KernelNodeParms<T>>& kernelParms)
{
    switch(ComputeGraphBuilder::current_phase())
    {
        case ComputeGraphPhase::TopoBuilding:
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


MUDA_INLINE void details::ComputeGraphAccessor::set_memcpy_node(void*       dst,
                                                                const void* src,
                                                                size_t size_bytes,
                                                                cudaMemcpyKind kind)
{
    switch(ComputeGraphBuilder::current_phase())
    {
        case ComputeGraphPhase::TopoBuilding:
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

MUDA_INLINE details::ComputeGraphAccessor::ComputeGraphAccessor()
    : m_cg(*ComputeGraphBuilder::current_graph())
{
}

MUDA_INLINE void details::ComputeGraphAccessor::check_allow_var_eval() const
{
    if(m_cg.m_is_in_capture_func)
        MUDA_ERROR_WITH_LOCATION("you can't eval a var in ComputeGraph::capture() function");
}

MUDA_INLINE void details::ComputeGraphAccessor::check_allow_node_adding() const
{
    if(m_cg.current_graph_phase() != ComputeGraphPhase::None)
        MUDA_ERROR_WITH_LOCATION("you are not allowed adding node at this point");
}

MUDA_INLINE void details::ComputeGraphAccessor::set_var_usage(VarId id, ComputeGraphVarUsage usage)
{
    auto& dst_usage = m_cg.m_temp_node_info.var_usage[id];
    if(dst_usage < usage)
        dst_usage = usage;
}

namespace details
{
    MUDA_INLINE void process_node(std::vector<ComputeGraph::Dependency>& deps,
                                  std::vector<NodeId>& last_read_or_write_nodes,
                                  std::vector<NodeId>& last_write_nodes,
                                  const std::vector<ComputeGraphVarBase*> input_vars,
                                  std::vector<ComputeGraphNodeBase*> nodes,
                                  NodeId    current_node_id,
                                  uint64_t& dep_begin,
                                  uint64_t& dep_count)
    {
        auto is_read_write = [](ComputeGraphVarUsage usage)
        { return usage == ComputeGraphVarUsage::ReadWrite; };
        auto is_read_only = [](ComputeGraphVarUsage usage)
        { return usage == ComputeGraphVarUsage::Read; };

        std::unordered_set<VarId::type> unique_deps;
        auto                            node = nodes[current_node_id.value()];

        for(auto& [arg_id, usage] : node->var_usages())
        {
            // if this is a written resource,
            // this should depend on any write and read before it
            // to get newest data or to avoid data corruption
            if(is_read_write(usage))
            {
                auto dst_nid = last_read_or_write_nodes[arg_id.value()];
                if(dst_nid.is_valid())
                {
                    // the last accessing node reads or writes this resrouce, so I should depend on it
                    if(unique_deps.find(dst_nid.value()) == unique_deps.end())
                    {
                        // record this dependency
                        unique_deps.insert(dst_nid.value());
                    }
                }
            }
            // if this is a read resource,
            // this should depend on any write before it
            // to get newest data
            // but it has no need to depend on any read before it
            else if(is_read_only(usage))
            {
                auto dst_nid = last_write_nodes[arg_id.value()];
                if(dst_nid.is_valid())
                {
                    // the last accessing node writes this resrouce, so I should depend on it
                    if(unique_deps.find(dst_nid.value()) == unique_deps.end())
                    {
                        // record this dependency
                        unique_deps.insert(dst_nid.value());
                    }
                }
            }
        }

        // set up res node map with pair [res, node]
        for(auto& [arg_id, usage] : node->var_usages())
        {
            // if this is a write resource,
            // the latter read/write kernel should depend on this
            // to get the newest data.
            if(is_read_write(usage))
            {
                last_read_or_write_nodes[arg_id.value()] = current_node_id;
                last_write_nodes[arg_id.value()]         = current_node_id;
            }
            // if this is a read resource,
            // the latter write kernel should depend on this
            // to avoid data corruption.
            else if(is_read_only(usage))
            {
                last_read_or_write_nodes[arg_id.value()] = current_node_id;
            }
        }

        // add dependencies to deps
        dep_begin = deps.size();
        for(auto dep : unique_deps)
            deps.emplace_back(ComputeGraph::Dependency{current_node_id, NodeId{dep}});
        dep_count = unique_deps.size();
    }
}  // namespace details

MUDA_INLINE void ComputeGraph::cuda_graph_add_deps()
{
    std::vector<cudaGraphNode_t> from;
    from.reserve(m_deps.size());
    std::vector<cudaGraphNode_t> to;
    to.reserve(m_deps.size());

    std::for_each(m_deps.begin(),
                  m_deps.end(),
                  [&](const Dependency& dep)
                  {
                      from.emplace_back(m_nodes[dep.dst.value()]->handle());
                      to.emplace_back(m_nodes[dep.src.value()]->handle());
                  });

    checkCudaErrors(cudaGraphAddDependencies(
        m_graph.handle(), from.data(), to.data(), m_deps.size()));
}

MUDA_INLINE void ComputeGraph::build_deps()
{
    m_deps.clear();
    // map: var_id -> node_id, uint64_t{-1} means no write node yet
    auto last_write_nodes = std::vector<NodeId>(m_vars.size());
    // map: var_id -> node_id, uint64_t{-1} means no read node yet
    auto last_read_nodes = std::vector<NodeId>(m_vars.size());

    // process all nodes
    for(size_t i = 0u; i < m_nodes.size(); i++)
    {
        size_t dep_begin, dep_count;
        details::process_node(
            m_deps, last_read_nodes, last_write_nodes, m_vars, m_nodes, NodeId{i}, dep_begin, dep_count);
        m_nodes[i]->set_deps_range(dep_begin, dep_count);
    }

    m_is_topo_built = true;
}
}  // namespace muda