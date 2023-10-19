#pragma once
#include <muda/launch/launch_base.h>

namespace muda
{
class ComputeGraph;
class ComputeGraphLaunch : public LaunchBase<ComputeGraphLaunch>
{
  public:
    ComputeGraphLaunch(cudaStream_t stream = nullptr)
        : LaunchBase(stream)
    {
    }
    ~ComputeGraphLaunch() = default;

    ComputeGraphLaunch launch(ComputeGraph& graph) const;
};
}  // namespace muda

#include "details/compute_graph_launch.inl"