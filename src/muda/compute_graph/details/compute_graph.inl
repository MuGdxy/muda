#pragma once
#include <memory>

#include <muda/exception.h>
#include <muda/compute_graph/compute_graph_var.h>
#include <muda/compute_graph/compute_graph_node.h>
#include <muda/compute_graph/nodes/compute_graph_kernel_node.h>

namespace muda
{
template <typename T>
MUDA_INLINE ComputeGraphVar<T>* ComputeGraph::create_var(std::string_view name)
{
    auto ptr = new ComputeGraphVar<T>(this, name, VarId{m_vars.size()});
    m_vars.emplace_back(ptr);
    m_vars_map.emplace(name, ptr);
    return ptr;
}
template <typename T>
inline ComputeGraphVar<T>* ComputeGraph::create_var(std::string_view name, T&& init_value)
{
    auto ptr =
        new ComputeGraphVar<T>(this, name, VarId{m_vars.size()}, std::forward<T>(init_value));
    m_vars.emplace_back(ptr);
    m_vars_map.emplace(name, ptr);
    return ptr;
}
template <typename T>
MUDA_INLINE ComputeGraphVar<T>* ComputeGraph::find_var(std::string_view name)
{
    auto it = m_vars_map.find(std::string{name});
    if(it == m_vars_map.end())
        return nullptr;
    return dynamic_cast<ComputeGraphVar<T>*>(it->second);
};

MUDA_INLINE void ComputeGraph::build()
{
    if(m_graph_exec)
        return;

    ComputeGraphBuilder::current_graph(this);
    m_current_graph_phase = ComputeGraphPhase::Building;
    m_closure_need_update.resize(m_closures.size(), false);
    for(size_t i = 0; i < m_closures.size(); ++i)
    {
        m_current_node_id    = NodeId{i};
        m_current_closure_id = ClosureId{i};
        m_allow_access_graph = true;
        m_closures[i].second();
    }

    m_graph_exec          = m_graph.instantiate();
    m_current_graph_phase = ComputeGraphPhase::None;
}

MUDA_INLINE void ComputeGraph::check_vars_valid()
{
    for(auto& var : m_vars)
        if(!var->is_valid())
            throw muda::logic_error("var[" + std::string{var->name()} + "] is not valid");
}

MUDA_INLINE void ComputeGraph::_update()
{
    if(!m_need_update)
        return;

    m_current_graph_phase = ComputeGraphPhase::Updating;
    for(size_t i = 0; i < m_closure_need_update.size(); ++i)
    {
        auto& need_update = m_closure_need_update[i];
        if(need_update)
        {
            m_current_closure_id = ClosureId{i};
            m_current_node_id    = NodeId{i};
            m_allow_access_graph = true;
            m_closures[i].second();
            need_update = false;
        }
    }
    m_current_graph_phase = ComputeGraphPhase::None;
}

MUDA_INLINE void ComputeGraph::launch(cudaStream_t s)
{
    m_allow_node_adding = false;
    check_vars_valid();
    build();
    _update();
    m_graph_exec->launch(s);
}

template <typename T>
MUDA_INLINE void details::ComputeGraphAccessor::add_kernel_node(const S<KernelNodeParms<T>>& parms)
{
    access_graph([&](Graph& g) {  // create kernel node
        const auto& [name, closure] = current_closure();

        auto kernel_node =                    //
            new ComputeGraphKernelNode(       //
                &m_cg,                        // compute graph
                name,                         // node name
                NodeId{m_cg.m_nodes.size()},  // node id
                temp_var_usage(),             // var usage
                g.add_kernel_node(parms));    // kernel node
        m_cg.m_nodes.emplace_back(kernel_node);
    });
}

template <typename T>
MUDA_INLINE void details::ComputeGraphAccessor::update_kernel_node(const S<KernelNodeParms<T>>& kernelParms)
{
    access_graph_exec(
        [&](GraphExec& g_exec)
        {
            const auto& [name, closure] = current_closure();
            auto node                   = current_node();
            auto kernel_node = dynamic_cast<ComputeGraphKernelNode*>(node);
            g_exec.set_kernel_node_parms(kernel_node->m_node, kernelParms);
        });
}

MUDA_INLINE void details::ComputeGraphAccessor::set_var_usage(VarId id, ComputeGraphVarUsage usage)
{
    auto& dst_usage = m_cg.m_temp_node_info.var_usage[id];
    if(dst_usage < usage)
        dst_usage = usage;
}
}  // namespace muda