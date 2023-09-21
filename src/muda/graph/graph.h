#pragma once
#include <unordered_map>
#include <unordered_set>

#include "base.h"

#include "kernel_node.h"
#include "memory_node.h"
#include "host_node.h"
#include "graph_exec.h"
#include "event_node.h"

namespace muda
{
class Graph
{
    template <typename T>
    using S = std::shared_ptr<T>;
    template <typename T>
    using U = std::unique_ptr<T>;
    cudaGraph_t m_handle;

  public:
    Graph() { checkCudaErrors(cudaGraphCreate(&m_handle, 0)); }

    friend class GraphExec;
    friend class std::shared_ptr<Graph>;

    MUDA_NODISCARD S<GraphExec> instantiate()
    {
        auto ret = std::make_shared<GraphExec>();
        checkCudaErrors(cudaGraphInstantiate(&ret->m_handle, m_handle, nullptr, nullptr, 0));
        return ret;
    }

    template <typename T>
    S<KernelNode> add_kernel_node(const S<KernelNodeParms<T>>&     kernelParms,
                                  const std::vector<S<GraphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<KernelNode>();
        std::vector<cudaGraphNode_t> nodes = map_dependencies(deps);
        checkCudaErrors(cudaGraphAddKernelNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), kernelParms->handle()));
        return ret;
    }

    template <typename T>
    S<HostNode> add_host_node(const S<HostNodeParms<T>>&       hostParms,
                            const std::vector<S<GraphNode>>& deps = {})
    {
        cached.push_back(hostParms);
        auto                         ret   = std::make_shared<HostNode>();
        std::vector<cudaGraphNode_t> nodes = map_dependencies(deps);
        checkCudaErrors(cudaGraphAddHostNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), hostParms->handle()));
        return ret;
    }

    template <typename T>
    auto add_memcpy_node(T*                               dst,
                       const T*                         src,
                       size_t                           count,
                       cudaMemcpyKind                   kind,
                       const std::vector<S<GraphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<memcpyNode>();
        std::vector<cudaGraphNode_t> nodes = map_dependencies(deps);
        checkCudaErrors(cudaGraphAddMemcpyNode1D(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), dst, src, sizeof(T) * count, kind));
        return ret;
    }

    auto add_event_record_node(cudaEvent_t e, const std::vector<S<GraphNode>>& deps = {})
    {
        auto                         ret = std::make_shared<EventRecordNode>();
        std::vector<cudaGraphNode_t> nodes = map_dependencies(deps);
        checkCudaErrors(cudaGraphAddEventRecordNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), e));
        return ret;
    }

    auto add_event_wait_node(cudaEvent_t e, const std::vector<S<GraphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<EventWaitNode>();
        std::vector<cudaGraphNode_t> nodes = map_dependencies(deps);
        checkCudaErrors(cudaGraphAddEventWaitNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), e));
        return ret;
    }

    auto add_dependency(S<GraphNode> from, S<GraphNode> to)
    {
        checkCudaErrors(
            cudaGraphAddDependencies(m_handle, &(from->m_handle), &(to->m_handle), 1));
    }

    ~Graph() { checkCudaErrors(cudaGraphDestroy(m_handle)); }

    static auto create() { return std::make_shared<Graph>(); }

  private:
    // keep the ref count > 0 for those whose data should be kept alive for the graph life.
    std::list<S<NodeParms>> cached;

    static std::vector<cudaGraphNode_t> map_dependencies(const std::vector<S<GraphNode>>& deps)
    {
        std::vector<cudaGraphNode_t> nodes;
        nodes.reserve(deps.size());
        for(auto d : deps)
            nodes.push_back(d->m_handle);
        return nodes;
    }
};


template <typename T>
MUDA_INLINE size_t make_resource_id(T& t) MUDA_NOEXCEPT
{
    return size_t(std::addressof(t));
}


class res
{
  public:
    enum class type
    {
        r = 1,
        w = 1 << 1
    };

  private:
    // resource ids
    std::vector<std::pair<size_t, type>> m_ids;
    friend class graphManager;

  public:
    static constexpr type r = type::r;
    static constexpr type w = type::w;

    template <typename... Args>
    res(Args&... args)
    {
        auto mode = w;
        m_ids.reserve(sizeof...(args));
        (process(mode, args), ...);
    }

  private:
    template <typename Arg>
    void process(type& mode, Arg& arg)
    {
        if constexpr(std::is_same_v<std::decay_t<Arg>, type>)  // mode change
        {
            mode = arg;
        }
        else  // set res
        {
            m_ids.push_back({make_resource_id(arg), mode});
        }
    }
};
}  // namespace muda