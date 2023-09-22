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

    static void invoke(const std::function<void()>& none,
                       const std::function<void()>& building,
                       const std::function<void()>& updating)
    {
        switch(current_phase())
        {
            case ComputeGraphPhase::None: {
                none();
            }
            break;
            case ComputeGraphPhase::Building: {
                building();
            }
            break;
            case ComputeGraphPhase::Updating: {
                updating();
            }
            break;
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