#pragma once
#include "base.h"

namespace muda
{
class KernelNode : public GraphNode
{
  public:
    using this_type = KernelNode;
    friend class Graph;
};

template <typename U>
class KernelNodeParms : public NodeParms
{
    std::vector<void*>   m_args;
    cudaKernelNodeParams m_parms;

  public:
    using this_type = KernelNodeParms;
    friend class Graph;
    friend class std::shared_ptr<this_type>;
    friend class std::unique_ptr<this_type>;
    friend class std::weak_ptr<this_type>;

    template <typename... Args>
    KernelNodeParms(Args&&... args)
        : kernelParmData(std::forward<Args>(args)...)
        , m_parms({})
    {
    }

    KernelNodeParms() {}
    U    kernelParmData;
    auto func() { return m_parms.func; }
    void func(void* v) { m_parms.func = v; }
    auto gridDim() { return m_parms.gridDim; }
    void gridDim(const dim3& v) { m_parms.gridDim = v; }
    auto blockDim() { return m_parms.blockDim; }
    void blockDim(const dim3& v) { m_parms.blockDim = v; }
    auto sharedMemBytes() { return m_parms.sharedMemBytes; }
    void sharedMemBytes(unsigned int v) { m_parms.sharedMemBytes = v; }
    auto kernelParams() { return m_parms.kernelParams; }
    void kernelParams(const std::vector<void*>& v)
    {
        m_args               = v;
        m_parms.kernelParams = m_args.data();
    }
    void parse(std::function<std::vector<void*>(U&)> pred)
    {
        m_args               = pred(kernelParmData);
        m_parms.kernelParams = m_args.data();
    }
    auto extra() { return m_parms.extra; }
    void extra(void** v) { m_parms.extra = v; }

    const cudaKernelNodeParams* handle() const { return &m_parms; }
};
}  // namespace muda