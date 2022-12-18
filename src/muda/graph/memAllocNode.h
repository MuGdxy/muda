#pragma once
#include "graph_base.h"

namespace muda
{
class memAllocNode : public graphNode
{
    void* dptr;

  public:
    using this_type = memAllocNode;
    friend class graph;
};

template <typename T>
class memAllocNodeParms
{
    cudaMemAllocNodeParams parms;

  public:
    using this_type = memAllocNodeParms;
    friend class graph;
    friend class std::shared_ptr<this_type>;
    friend class std::unique_ptr<this_type>;
    friend class std::weak_ptr<this_type>;

    memAllocNodeParms(size_t size)
        : parms({})
    {
        parms.poolProps.allocType = cudaMemAllocationTypePinned;
        cudaGetDevice(&parms.poolProps.location.id);
        parms.poolProps.location.type = cudaMemLocationTypeDevice;
        parms.bytesize                = size * sizeof(T);
    }

    cudaMemAllocNodeParams*       getRaw() { return &parms; }
    const cudaMemAllocNodeParams* getRaw() const { return &parms; }
};
}  // namespace muda