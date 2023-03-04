#pragma once
#include "base.h"

namespace muda
{
#ifdef MUDA_WITH_GRAPH_MEMORY_ALLOC_FREE
class memAllocNode : public graphNode
{
    void* m_dptr;

  public:
    using this_type = memAllocNode;
    friend class graph;
};

template <typename T>
class memAllocNodeParms : public nodeParms
{
    cudaMemAllocNodeParams m_parms;

  public:
    using this_type = memAllocNodeParms;
    friend class graph;
    friend class std::shared_ptr<this_type>;
    friend class std::unique_ptr<this_type>;
    friend class std::weak_ptr<this_type>;

    memAllocNodeParms(size_t size)
        : m_parms({})
    {
        m_parms.poolProps.allocType = cudaMemAllocationTypePinned;
        cudaGetDevice(&m_parms.poolProps.location.id);
        m_parms.poolProps.location.type = cudaMemLocationTypeDevice;
        m_parms.bytesize                = size * sizeof(T);
    }

    cudaMemAllocNodeParams*       getRaw() { return &m_parms; }
    const cudaMemAllocNodeParams* getRaw() const { return &m_parms; }
};

class memFreeNode : public graphNode
{
  public:
    using this_type = memFreeNode;
    friend class graph;
};
#endif

class memcpyNode : public graphNode
{
  public:
    using this_type = memcpyNode;
    friend class graph;
};
}  // namespace muda