#pragma once
#include <cuda.h>
#include <muda/viewer/viewer_base.h>
namespace muda
{
class GraphViewer : public ViewerBase
{
    MUDA_VIEWER_COMMON_NAME(GraphViewer);

  public:
    MUDA_GENERIC      GraphViewer(cudaGraphExec_t graph);
    MUDA_GENERIC void launch(cudaStream_t stream = nullptr) const;
    MUDA_GENERIC auto handle() const { return m_graph; }

    MUDA_DEVICE void  tail_launch() const;
    MUDA_DEVICE void  fire_and_forget() const;


  private:
    friend class ComputeGraph;
    cudaGraphExec_t m_graph;
};
}  // namespace muda

#include "details/graph_viewer.inl"