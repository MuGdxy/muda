#pragma once
#include "graph_base.h"
#include "kernelNode.h"
#include "memCpyNode.h"

namespace muda
{
class graphExec
{
    cudaGraphExec_t handle;

  public:
    friend class graph;

    graphExec()
        : handle(nullptr)
    {
    }

    void launch(cudaStream_t stream = nullptr)
    {
        checkCudaErrors(cudaGraphLaunch(handle, stream));
    }

    template <typename T>
    void setKernelNodeParms(sptr<kernelNode> node, const sptr<kernelNodeParms<T>>& new_parms)
    {
        checkCudaErrors(cudaGraphExecKernelNodeSetParams(
            handle, node.get()->handle, new_parms->getRaw()));
    }

    template <typename T>
    void setMemcpyNodeParms(sptr<memcpyNode> node, T* dst, const T* src, size_t count, cudaMemcpyKind kind)
    {
        checkCudaErrors(cudaGraphExecMemcpyNodeSetParams1D(
            handle, node.get()->handle, dst, src, count * sizeof(T), kind));
    }

    ~graphExec()
    {
        if(handle)
            checkCudaErrors(cudaGraphExecDestroy(handle));
    }
    //cudaGraphExec_t getRaw() const { return handle; }
};
}  // namespace muda