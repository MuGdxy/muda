#pragma once
#include <muda/graph/graph_base.h>
#include <muda/graph/kernel_node.h>
#include <muda/graph/memory_node.h>
#include <muda/graph/event_node.h>

namespace muda
{
class GraphExec
{
    template <typename T>
    using S = std::shared_ptr<T>;
    template <typename T>
    using U = std::unique_ptr<T>;
    cudaGraphExec_t m_handle;

  public:
    friend class Graph;

    GraphExec()
        : m_handle(nullptr)
    {
    }

    void launch(cudaStream_t stream = nullptr)
    {
        checkCudaErrors(cudaGraphLaunch(m_handle, stream));
    }

    template <typename T>
    void set_kernel_node_parms(S<KernelNode> node, const S<KernelNodeParms<T>>& new_parms)
    {
        checkCudaErrors(cudaGraphExecKernelNodeSetParams(
            m_handle, node->m_handle, new_parms->handle()));
    }

    void set_memcpy_node_parms(S<MemcpyNode> node, void* dst, const void* src, size_t size_bytes, cudaMemcpyKind kind)
    {
        checkCudaErrors(cudaGraphExecMemcpyNodeSetParams1D(
            m_handle, node->m_handle, dst, src, size_bytes, kind));
    }

    void set_event_record_node_parms(S<EventRecordNode> node, cudaEvent_t event)
    {
        checkCudaErrors(cudaGraphExecEventRecordNodeSetEvent(m_handle, node->m_handle, event));
    }

    void set_event_wait_node_parms(S<EventWaitNode> node, cudaEvent_t event)
    {
        checkCudaErrors(cudaGraphExecEventWaitNodeSetEvent(m_handle, node->m_handle, event));
    }

    ~GraphExec()
    {
        if(m_handle)
            checkCudaErrors(cudaGraphExecDestroy(m_handle));
    }

    cudaGraphExec_t handle() const { return m_handle; }

  private:
    // keep the ref count > 0 for those whose data should be kept alive for the graph life.
    std::list<S<NodeParms>> cached;
};
}  // namespace muda