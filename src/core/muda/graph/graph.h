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
class graph
{
    cudaGraph_t m_handle;

  public:
    graph() { checkCudaErrors(cudaGraphCreate(&m_handle, 0)); }

    friend class graphExec;
    friend class std::shared_ptr<graph>;

    MUDA_NODISCARD sptr<graphExec> instantiate()
    {
        auto ret = std::make_shared<graphExec>();
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

#ifdef MUDA_WITH_GRAPH_MEMORY_ALLOC_FREE
    template <typename T>
    auto addMemAllocNode(sptr<memAllocNodeParms<T>>&         memAllocParms,
                         const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         node  = std::make_shared<memAllocNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddMemAllocNode(
            &node->m_handle, m_handle, nodes.data(), nodes.size(), memAllocParms->getRaw()));
        auto ptr     = reinterpret_cast<T*>(memAllocParms->getRaw()->dptr);
        node->m_dptr = ptr;
        return std::make_tuple(node, ptr);
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

    auto addMemFreeNode(void* ptr, const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<memFreeNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddMemFreeNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), ptr));
        return ret;
    }
#endif

    auto addEventRecordNode(cudaEvent_t e, const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         ret = std::make_shared<eventRecordNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddEventRecordNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), e));
        return ret;
    }

    auto addEventWaitNode(cudaEvent_t e, const std::vector<sptr<graphNode>>& deps = {})
    {
        auto                         ret   = std::make_shared<eventWaitNode>();
        std::vector<cudaGraphNode_t> nodes = mapDependencies(deps);
        checkCudaErrors(cudaGraphAddEventWaitNode(
            &ret->m_handle, m_handle, nodes.data(), nodes.size(), e));
        return ret;
    }

    auto addDependency(sptr<graphNode> from, sptr<graphNode> to)
    {
        checkCudaErrors(
            cudaGraphAddDependencies(m_handle, &(from->m_handle), &(to->m_handle), 1));
    }

    ~graph() { checkCudaErrors(cudaGraphDestroy(m_handle)); }

    static auto create() { return std::make_shared<graph>(); }

  private:
    // keep the ref count > 0 for those whose data should be kept alive for the graph life.
    std::list<sptr<nodeParms>> cached;

    static std::vector<cudaGraphNode_t> mapDependencies(const std::vector<sptr<graphNode>>& deps)
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


class graphManager
{
    graph                                       m_graph;
    std::unordered_map<size_t, sptr<graphNode>> m_write_deps;
    std::unordered_map<size_t, sptr<graphNode>> m_read_deps;

  public:
    template <typename T>
    auto addKernelNode(const sptr<kernelNodeParms<T>>& kernelParms, const res& res)
    {
        auto node = m_graph.addKernelNode(kernelParms);
        addNode(node, res);
        return node;
    }

    auto addEventRecordNode(cudaEvent_t event, const res& res)
    {
        auto node = m_graph.addEventRecordNode(event);
        addNode(node, res);
        return node;
    }

    auto addEventWaitNode(cudaEvent_t event, const res& res)
    {
        auto node = m_graph.addEventWaitNode(event);
        addNode(node, res);
        return node;
    }

    MUDA_NODISCARD sptr<graphExec> instantiate()
    {
        return m_graph.instantiate();
    }

  private:
    void addNode(sptr<graphNode> node, const res& res)
    {
        // find write/read dependency
        std::unordered_set<cudaGraphNode_t> deps;
        for(auto [id, mode] : res.m_ids)
        {
            // if this is a write resource,
            // this should depend on any write and read before it
            // to get newest data or to avoid data corruption
            if(mode == res::w)
            {
                auto find = m_write_deps.find(id);
                if(find != m_write_deps.end())  // has write depend node
                {
                    auto handle = find->second->getRaw();
                    if(deps.find(handle) == deps.end())
                    {
                        deps.insert(handle);
                        m_graph.addDependency(find->second, node);
                    }
                }
            }
            // if this is a read resource,
            // this should depend on any write before it
            // to get newest data
            else if(mode == res::r)
            {
                auto find = m_read_deps.find(id);
                if(find != m_read_deps.end())  // has read depend node
                {
                    auto handle = find->second->getRaw();
                    if(deps.find(handle) == deps.end())
                    {
                        deps.insert(handle);
                        m_graph.addDependency(find->second, node);
                    }
                }
            }
        }

        // set up res node map with pair [res, node]
        for(auto [id, mode] : res.m_ids)
        {
            // if this is a write resource,
            // the latter read/write kernel should depend on this
            // to get the newest data.
            if(mode == res::w)
            {
                m_read_deps[id]  = node;
                m_write_deps[id] = node;
            }
            // if this is a read resource,
            // the latter write kernel should depend on this
            // to avoid data corruption.
            else if(mode == res::r)
            {
                m_write_deps[id] = node;
            }
        }
    }
};
}  // namespace muda