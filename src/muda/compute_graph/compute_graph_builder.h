#pragma once
#include <muda/compute_graph/compute_graph.h>
#include <muda/compute_graph/compute_graph_phase.h>

namespace muda::details
{
class ComputeGraphAccessor;
}
namespace muda
{
class ComputeGraph;
class ComputeGraphBuilder
{
    static ComputeGraphBuilder& instance();
    using Phase = ComputeGraphPhase;

  public:
    static Phase current_phase();
    static bool  is_phase_none();
    static bool  is_phase_serial_launching();
    static bool  is_topo_building();
    static bool  is_building();

    // no graph building or the graph is in serial launching mode
    static bool is_direct_launching();

  private:
    friend class ComputeGraph;

    static auto current_graph(ComputeGraph* graph);
    friend class details::ComputeGraphAccessor;
    static auto current_graph() { return instance().m_current_graph; }

    ComputeGraphBuilder()  = default;
    ~ComputeGraphBuilder() = default;

    ComputeGraph* m_current_graph = nullptr;
};
}  // namespace muda

#include <muda/compute_graph/details/compute_graph_builder.inl>