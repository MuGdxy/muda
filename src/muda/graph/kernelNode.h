#pragma once
#include "graph_base.h"

namespace muda
{
class kernelNode : public graphNode
{
  public:
    using this_type = kernelNode;
    friend class graph;
};

template <typename U>
class kernelNodeParms
{
    std::vector<void*>   args;
    cudaKernelNodeParams parms;

  public:
    using this_type = kernelNodeParms;
    friend class graph;
    friend class std::shared_ptr<this_type>;
    friend class std::unique_ptr<this_type>;
    friend class std::weak_ptr<this_type>;

    template <typename... Args>
    kernelNodeParms(Args&&... args)
        : kernelParmData(std::forward<Args>(args)...)
        , parms({})
    {
    }

    kernelNodeParms() {}
    U    kernelParmData;
    auto func() { return parms.func; }
    void func(void* v) { parms.func = v; }
    auto gridDim() { return parms.gridDim; }
    void gridDim(const dim3& v) { parms.gridDim = v; }
    auto blockDim() { return parms.blockDim; }
    void blockDim(const dim3& v) { parms.blockDim = v; }
    auto sharedMemBytes() { return parms.sharedMemBytes; }
    void sharedMemBytes(unsigned int v) { parms.sharedMemBytes = v; }
    auto kernelParams() { return parms.kernelParams; }
    void kernelParams(const std::vector<void*>& v)
    {
        args               = v;
        parms.kernelParams = args.data();
    }
    void parse(std::function<std::vector<void*>(U&)> pred)
    {
        args               = pred(kernelParmData);
        parms.kernelParams = args.data();
    }
    auto extra() { return parms.extra; }
    void extra(void** v) { parms.extra = v; }

    const cudaKernelNodeParams* getRaw() const { return &parms; }
};
}  // namespace muda