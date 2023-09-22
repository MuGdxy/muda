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

inline void ComputeGraphVarBase::_building_eval(ComputeGraphVarUsage usage) const
{
    m_closure_ids.insert(m_graph->current_closure_id());
    details::ComputeGraphAccessor().set_var_usage(var_id(), usage);
}


template <typename T>
MUDA_INLINE ComputeGraphVar<T>::RWViewer ComputeGraphVar<T>::eval()
{
    auto phase = ComputeGraphBuilder::current_phase();
    switch(phase)
    {
        case ComputeGraphPhase::None: {
            throw std::logic_error("ComputeGraphVar.eval() is not allowed outside Graph Closure");
        }
        break;
        case ComputeGraphPhase::Building: {
            this->base_building_eval();
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
MUDA_INLINE ComputeGraphVar<T>::ROViewer ComputeGraphVar<T>::ceval() const
{
    auto phase = ComputeGraphBuilder::current_phase();
    switch(phase)
    {
        case ComputeGraphPhase::None: {
            throw std::logic_error("ComputeGraphVar.eval() is not allowed outside Graph Closure");
        }
        break;
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