namespace muda
{
MUDA_INLINE ComputeGraphViewer::ComputeGraphViewer(cudaGraphExec_t graph)
    : m_graph(graph)
{
}
MUDA_INLINE MUDA_GENERIC void ComputeGraphViewer::launch(cudaStream_t stream)
{
    checkCudaErrors(cudaGraphLaunch(m_graph, stream));
}
}
