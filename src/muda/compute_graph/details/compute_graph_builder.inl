#pragma once

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

MUDA_INLINE bool ComputeGraphBuilder::is_phase_none()
{
    return current_phase() == ComputeGraphPhase::None;
}

MUDA_INLINE bool ComputeGraphBuilder::is_phase_serial_launching()
{
    return current_phase() == ComputeGraphPhase::SerialLaunching;
}

MUDA_INLINE bool ComputeGraphBuilder::is_direct_launching()
{
    return is_phase_serial_launching() || is_phase_none();
}

MUDA_INLINE auto ComputeGraphBuilder::current_graph(ComputeGraph* graph)
{
    return instance().m_current_graph = graph;
}

MUDA_INLINE ComputeGraphBuilder& muda::ComputeGraphBuilder::instance()
{
    thread_local static ComputeGraphBuilder builder;
    return builder;
}
}  // namespace muda