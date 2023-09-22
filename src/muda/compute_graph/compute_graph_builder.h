#pragma once
#include <muda/compute_graph/compute_graph_phase.h>
namespace muda
{
class ComputeGraph;
class ComputeGraphBuilder
{
  public:
    static auto current_graph() { return instance().m_current_graph; }

    static ComputeGraphPhase current_phase();

    template <typename F1, typename F2, typename F3>
    static void invoke(F1&& none, F2&& building, F3&& updating)
    {
        switch(current_phase())
        {
            case ComputeGraphPhase::None:
                none();
                break;
            case ComputeGraphPhase::Building:
                building();
                break;
            case ComputeGraphPhase::Updating:
                updating();
            default:
                break;
        }
    }

  private:
    friend class ComputeGraph;
    static auto current_graph(ComputeGraph* graph)
    {
        return instance().m_current_graph = graph;
    }

    ComputeGraphBuilder()  = default;
    ~ComputeGraphBuilder() = default;

    ComputeGraph* m_current_graph = nullptr;

    static ComputeGraphBuilder& instance()
    {
        thread_local static ComputeGraphBuilder builder;
        return builder;
    }
};
}  // namespace muda

#include <muda/compute_graph/details/compute_graph_builder.inl>