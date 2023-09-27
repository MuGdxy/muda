#pragma once
#include <muda/compute_graph/compute_graph.h>
#include <muda/compute_graph/compute_graph_node.h>

namespace muda
{
MUDA_INLINE void ComputeGraphVarBase::base_update()
{
    m_graph->m_need_update = true;
    for(auto& id : m_closure_ids)
        m_graph->m_closure_need_update[id.value()] = true;
    m_is_valid = true;
}
MUDA_INLINE void ComputeGraphVarBase::base_building_eval()
{
    _building_eval(ComputeGraphVarUsage::ReadWrite);
}

MUDA_INLINE void ComputeGraphVarBase::base_building_eval_const() const
{
    _building_eval(ComputeGraphVarUsage::Read);
}

MUDA_INLINE void ComputeGraphVarBase::_building_eval(ComputeGraphVarUsage usage) const
{
    m_closure_ids.insert(m_graph->current_closure_id());
    details::ComputeGraphAccessor().set_var_usage(var_id(), usage);
}

MUDA_INLINE void ComputeGraphVarBase::graphviz_def(std::ostream& o) const
{
    graphviz_id(o);
    o << "[";
    if(!name().empty())
        o << "label=\"" << name() << "\",";
    o << R"(shape="rectangle", color="#F08705", style="filled,rounded", fillcolor="#F5AF58"])";
}

MUDA_INLINE void ComputeGraphVarBase::graphviz_id(std::ostream& o) const
{
    o << "var" << var_id();
}

template <typename T>
MUDA_INLINE typename ComputeGraphVar<T>::RWViewer ComputeGraphVar<T>::eval()
{
    auto phase = ComputeGraphBuilder::current_phase();
    switch(phase)
    {
        case ComputeGraphPhase::None: {
            throw std::logic_error("ComputeGraphVar.eval() is not allowed outside Graph Closure");
        }
        break;
        case ComputeGraphPhase::TopoBuilding:
        case ComputeGraphPhase::Building: {
            auto acc = details::ComputeGraphAccessor();
            acc.check_allow_var_eval();
            if (!acc.is_topo_built())
            {
                if constexpr(std::is_same_v<T, read_only_viewer_t<T>>)
                {
                    // they are all read only(e.g. host float/int ...)
                    this->base_building_eval_const();
                }
                else
                {
                    this->base_building_eval();
                }
            }
        }
        break;
        case ComputeGraphPhase::Updating:
        default:  // nothing to do
            break;
    }
    return m_value;
}

template <typename T>
MUDA_INLINE typename ComputeGraphVar<T>::ROViewer ComputeGraphVar<T>::ceval() const
{
    auto phase = ComputeGraphBuilder::current_phase();
    switch(phase)
    {
        case ComputeGraphPhase::None: {
            throw std::logic_error("ComputeGraphVar.eval() is not allowed outside Graph Closure");
        }
        break;
        case ComputeGraphPhase::TopoBuilding:
        case ComputeGraphPhase::Building: {
            this->base_building_eval_const();
        }
        break;
        case ComputeGraphPhase::Updating: {
            // nothing to do
        }
        default:
            break;
    }
    return m_value;
}
template <typename T>
MUDA_INLINE void muda::ComputeGraphVar<T>::update(const RWViewer& view)
{
    this->base_update();
    m_value = view;
}
}  // namespace muda