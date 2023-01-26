#pragma once
#include "base.h"
#include "kernel_node.h"
#include "memory_node.h"

namespace muda
{
class graphExec
{
    cudaGraphExec_t m_handle;

  public:
    friend class graph;

    graphExec()
        : m_handle(nullptr)
    {
    }

    void launch(cudaStream_t stream = nullptr)
    {
        checkCudaErrors(cudaGraphLaunch(m_handle, stream));
    }

    template <typename T>
    void setKernelNodeParms(sptr<kernelNode> node, const sptr<kernelNodeParms<T>>& new_parms)
    {
        checkCudaErrors(cudaGraphExecKernelNodeSetParams(
            m_handle, node.get()->m_handle, new_parms->getRaw()));
    }

    template <typename T>
    void setMemcpyNodeParms(sptr<memcpyNode> node, T* dst, const T* src, size_t count, cudaMemcpyKind kind)
    {
        checkCudaErrors(cudaGraphExecMemcpyNodeSetParams1D(
            m_handle, node.get()->m_handle, dst, src, count * sizeof(T), kind));
    }

    ~graphExec()
    {
        if(m_handle)
            checkCudaErrors(cudaGraphExecDestroy(m_handle));
    }
    //cudaGraphExec_t getRaw() const { return handle; }
  private:
    // keep the ref count > 0 for those whose data should be kept alive for the graph life.
    std::list<sptr<nodeParms>> cached;
};
}  // namespace muda