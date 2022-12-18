#pragma once
#include "graph_base.h"
#include "kernelNode.h"
#include "memAllocNode.h"
#include "memCpyNode.h"
#include "memFreeNode.h"
#include "hostNode.h"
#include "graphExec.h"
namespace muda
{
class graph
{
    cudaGraph_t handle;

  public:
    graph() { checkCudaErrors(cudaGraphCreate(&handle, 0)); }

    friend class graphExec;
    friend class std::shared_ptr<graph>;

    [[nodiscard]] sptr<graphExec> instantiate()
    {
        cudaGraphExec_t exec;
        auto            ret = std::make_shared<graphExec>();
        checkCudaErrors(cudaGraphInstantiate(&ret->handle, handle, nullptr, nullptr, 0));
        return ret;
    }

    template <typename T>
    sptr<kernelNode> addKernelNode(const sptr<kernelNodeParms<T>>& kernelParms,
                                   const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<kernelNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddKernelNode(
            &ret->handle, handle, nodes.data(), nodes.size(), kernelParms->getRaw()));
        return ret;
    }

    template <typename T>
    sptr<hostNode> addHostNode(const sptr<hostNodeParms<T>>&       hostParms,
                               const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<hostNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddHostNode(
            &ret->handle, handle, nodes.data(), nodes.size(), hostParms->getRaw()));
        return ret;
    }

    template <typename T>
    auto addMemAllocNode(sptr<memAllocNodeParms<T>>&         memAllocParms,
                         const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         node  = std::make_shared<memAllocNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddMemAllocNode(
            &node->handle, handle, nodes.data(), nodes.size(), memAllocParms->getRaw()));
        auto ptr   = reinterpret_cast<T*>(memAllocParms->getRaw()->dptr);
        node->dptr = ptr;
        return std::make_tuple(node, ptr);
    }

    template <typename T>
    auto addMemcpyNode(T*                                  dst,
                       const T*                            src,
                       size_t                              count,
                       cudaMemcpyKind                      kind,
                       const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<memcpyNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddMemcpyNode1D(
            &ret->handle, handle, nodes.data(), nodes.size(), dst, src, sizeof(T) * count, kind));
        return ret;
    }

    auto addMemFreeNode(sptr<memAllocNode>                  allocNode,
                        const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<memFreeNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddMemFreeNode(
            &ret->handle, handle, nodes.data(), nodes.size(), allocNode->dptr));
        return ret;
    }

    ~graph() { checkCudaErrors(cudaGraphDestroy(handle)); }

    static auto create() { return std::make_shared<graph>(); }

  private:
    //
    static std::vector<cudaGraphNode_t> mapDependencies(
        const std::vector<std::shared_ptr<graphNode>>& deps)
    {
        std::vector<cudaGraphNode_t> nodes;
        nodes.reserve(deps.size());
        for(auto d : deps)
            nodes.push_back(d->handle);
        return nodes;
    }
};
}  // namespace muda