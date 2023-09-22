#pragma once
#include <muda/compute_graph/compute_graph.h>
namespace muda
{
MUDA_INLINE ComputeGraphPhase ComputeGraphBuilder::current_phase()
{
    auto ins = instance().m_current_graph;
    if(ins)
        return ins->current_graph_phase();
    else
        return ComputeGraphPhase::None;
}
}  // namespace muda