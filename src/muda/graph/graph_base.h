#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

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

class graphNode
{
  protected:
    cudaGraphNode_t handle;

  public:
    friend class graphExec;
    graphNode()
        : handle(nullptr)
    {
    }
    using this_type = graphNode;
    friend class graph;
};
}  // namespace muda