#include <muda/compute_graph/compute_graph.h>

namespace muda
{
MUDA_INLINE ComputeGraphLaunch ComputeGraphLaunch::launch(ComputeGraph& graph) const
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(), 
        "ComputeGraphLaunch::launch() can't be called inside a graph");
    graph.launch(m_stream);
    return *this;
}
}  // namespace muda