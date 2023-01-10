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
    cudaGraph_t m_handle;

  public:
    graph() { checkCudaErrors(cudaGraphCreate(&m_handle, 0)); }

    friend class graphExec;
    friend class std::shared_ptr<graph>;

    [[nodiscard]] sptr<graphExec> instantiate()
    {
        auto            ret = std::make_shared<graphExec>();
        checkCudaErrors(cudaGraphInstantiate(&ret->m_handle, m_handle, nullptr, nullptr, 0));
        return ret;
    }

    template <typename T>
    sptr<kernelNode> addKernelNode(const sptr<kernelNodeParms<T>>& kernelParms,
                                   const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<kernelNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddKernelNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), kernelParms->getRaw()));
        return ret;
    }

    template <typename T>
    sptr<hostNode> addHostNode(const sptr<hostNodeParms<T>>&       hostParms,
                               const std::vector<sptr<graphNode>>& deps = {})
    {
        cached.push_back(hostParms);
        auto                         ret   = std::make_shared<hostNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddHostNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), hostParms->getRaw()));
        return ret;
    }

    template <typename T>
    auto addMemAllocNode(sptr<memAllocNodeParms<T>>&         memAllocParms,
                         const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         node  = std::make_shared<memAllocNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddMemAllocNode(
            &node->m_handle, m_handle, nodes.data(), nodes.size(), memAllocParms->getRaw()));
        auto ptr   = reinterpret_cast<T*>(memAllocParms->getRaw()->dptr);
        node->m_dptr = ptr;
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
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), dst, src, sizeof(T) * count, kind));
        return ret;
    }

    auto addMemFreeNode(sptr<memAllocNode>                  allocNode,
                        const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<memFreeNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddMemFreeNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), allocNode->m_dptr));
        return ret;
    }

    ~graph() { checkCudaErrors(cudaGraphDestroy(m_handle)); }

    static auto create() { return std::make_shared<graph>(); }

  private:
    // keep the ref count > 0 for those whose data should be kept alive for the graph life.
    std::list<sptr<nodeParms>> cached;
	
    //
    static std::vector<cudaGraphNode_t> mapDependencies(
        const std::vector<std::shared_ptr<graphNode>>& deps)
    {
        std::vector<cudaGraphNode_t> nodes;
        nodes.reserve(deps.size());
        for(auto d : deps)
            nodes.push_back(d->m_handle);
        return nodes;
    }
};
}  // namespace muda