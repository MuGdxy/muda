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
    using Phase       = ComputeGraphPhase;
    using PhaseAction = std::function<void()>;

  public:
    static Phase current_phase();
    static bool  is_phase_none();
    static bool  is_phase_serial_launching();
    static bool  is_topo_building();
    static bool  is_building();
    // return true when no graph is building or the graph is in serial launching mode
    static bool is_direct_launching();


    // do_when_direct_launch
    // do_when_set_node => do_when_add_node & do_when_update_node
    // if do_when_topo_building_set_node == nullptr, do_when_set_node will be called
    // if do_when_topo_building_set_node != nullptr, do_when_topo_building_set_node will be called
    static void invoke_phase_actions(PhaseAction&& do_when_direct_launch,
                                     PhaseAction&& do_when_set_node, 
                                     PhaseAction&& do_when_topo_building_set_node = nullptr);

  private:
    friend class ComputeGraph;
    friend class ComputeGraphVarBase;

    static auto current_graph(ComputeGraph* graph);
    friend class details::ComputeGraphAccessor;
    static auto current_graph() { return instance().m_current_graph; }

    ComputeGraphBuilder()  = default;
    ~ComputeGraphBuilder() = default;

    ComputeGraph* m_current_graph = nullptr;
};
}  // namespace muda

#include <muda/compute_graph/details/compute_graph_builder.inl>