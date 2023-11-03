#pragma once
#include <cuda.h>
#include <muda/viewer/viewer_base.h>
namespace muda
{
class GraphViewer : public ViewerBase
{
    MUDA_VIEWER_COMMON_NAME(GraphViewer);

  public:
    MUDA_HOST void launch(cudaStream_t stream = nullptr) const;
    MUDA_HOST      GraphViewer(cudaGraphExec_t graph);
    MUDA_HOST auto handle() const { return m_graph; }

  private:
    friend class ComputeGraph;
    cudaGraphExec_t m_graph;
};
}  // namespace muda

#include "details/graph_viewer.inl"