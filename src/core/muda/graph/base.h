#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include <list>
#include <vector>
#include <memory>
#include <functional>

#include "../check/checkCudaErrors.h"

namespace muda
{
template <typename T>
using sptr = std::shared_ptr<T>;
template <typename T>
using uptr = std::unique_ptr<T>;
template <typename T>
using wptr = std::weak_ptr<T>;

class graph;
class graphExec;

class nodeParms
{
  public:
    nodeParms()          = default;
    virtual ~nodeParms() = default;
};

class graphNode
{
  protected:
    cudaGraphNode_t m_handle;

  public:
    friend class graphExec;
    graphNode()
        : m_handle(nullptr)
    {
    }
    using this_type = graphNode;
    friend class graph;
    cudaGraphNode_t getRaw() const { return m_handle; }
};
}  // namespace muda