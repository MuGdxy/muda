#pragma once
#include <memory>
#include <muda/compute_graph/compute_graph_var.h>
#include <muda/compute_graph/compute_graph_node.h>

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
MUDA_INLINE ComputeGraphVar<T>* ComputeGraph::get_var(std::string_view name)
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
MUDA_INLINE void ComputeGraph::update()
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
            m_closures[i].second();
            need_update = false;
        }
    }
    m_current_graph_phase = ComputeGraphPhase::None;
}


//template <typename T>
//ComputeGraphNode<T>* ComputeGraph::create_node()
//{
//    auto id = current_closure_id().value();
//    auto closure_name = std::string_view{m_closures[id].first};
//    auto ptr = new ComputeGraphNode<T>(this, closure_name, NodeId{m_nodes.size()});
//    m_nodes.emplace_back(ptr);
//    return ptr;
//}

//MUDA_INLINE ComputeGraph::~ComputeGraph()
//{
//    for(auto v : m_vars)
//        delete v;
//    for(auto n : m_nodes)
//        delete n;
//}

}  // namespace muda