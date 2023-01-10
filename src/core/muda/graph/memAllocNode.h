#pragma once
#include "graph_base.h"

namespace muda
{
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
}  // namespace muda