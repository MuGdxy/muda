#pragma once
#include "graph_base.h"

namespace muda
{
class hostNode : public graphNode
{
  public:
    using this_type = hostNode;
    friend class graph;
};

template <typename T>
class hostNodeParms
{
    cudaHostNodeParams parms;

  public:
    T* hostData;
    using this_type = hostNodeParms;
    friend class graph;
    friend class std::shared_ptr<this_type>;
    friend class std::unique_ptr<this_type>;
    friend class std::weak_ptr<this_type>;

    hostNodeParms(T* f)
        : hostData(f)
        , parms({})
    {
        parms.userData = hostData;
    }
    auto fn() const { return parms.fn; }
    void fn(cudaHostFn_t fn) { parms.fn = fn; }
    auto userdata() const { return parms.userData; }
    void userdata(void* userdata) { parms.userData = userdata; }
    const cudaHostNodeParams* getRaw() const { return &parms; }
    cudaHostNodeParams*       getRaw() { return &parms; }
};
}  // namespace muda