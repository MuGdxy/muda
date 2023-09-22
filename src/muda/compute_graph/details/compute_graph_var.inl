#pragma once
#include <muda/compute_graph/compute_graph.h>

namespace muda
{
MUDA_INLINE void ComputeGraphVarBase::base_update()
{
    m_graph->m_need_update = true;
    for(auto& id : m_closure_ids)
        m_graph->m_closure_need_update[id.value()] = true;
}
MUDA_INLINE void muda::ComputeGraphVarBase::base_building_eval()
{
    m_closure_ids.insert(m_graph->current_closure_id());
}

MUDA_INLINE void ComputeGraphVarBase::base_building_eval_const() const
{
    m_closure_ids.insert(m_graph->current_closure_id());
}


template <typename T>
MUDA_INLINE T& ComputeGraphVar<T>::eval()
{
    auto phase = ComputeGraphBuilder::current_phase();
    switch(phase)
    {
        case ComputeGraphPhase::None: {
            throw std::logic_error("ComputeGraphVar.eval() is not allowed outside Graph Closure");
        }
        break;
        case ComputeGraphPhase::Building: {
            base_building_eval();
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
inline const T& ComputeGraphVar<T>::ceval() const
{
    auto phase = ComputeGraphBuilder::current_phase();
    switch(phase)
    {
        case ComputeGraphPhase::None: {
            throw std::logic_error("ComputeGraphVar.eval() is not allowed outside Graph Closure");
        }
        break;
        case ComputeGraphPhase::Building: {
            base_building_eval_const();
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
}  // namespace muda