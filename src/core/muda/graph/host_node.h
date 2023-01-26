#pragma once
#include "base.h"

namespace muda
{
class hostNode : public graphNode
{
  public:
    using this_type = hostNode;
    friend class graph;
};

template <typename T>
class hostNodeParms : public nodeParms
{
    cudaHostNodeParams m_parms;

  public:
    T hostData;
    using this_type = hostNodeParms;
    friend class graph;
    friend class std::shared_ptr<this_type>;
    friend class std::unique_ptr<this_type>;
    friend class std::weak_ptr<this_type>;

    hostNodeParms(const T& f)
        : hostData(f)
        , m_parms({})
    {
        m_parms.userData = &hostData;
    }
    auto fn() const { return m_parms.fn; }
    void fn(cudaHostFn_t fn) { m_parms.fn = fn; }
    auto userdata() const { return m_parms.userData; }
    void userdata(void* userdata) { m_parms.userData = userdata; }
    const cudaHostNodeParams* getRaw() const { return &m_parms; }
    cudaHostNodeParams*       getRaw() { return &m_parms; }
};
}  // namespace muda