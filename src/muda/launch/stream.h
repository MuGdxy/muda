#pragma once
#include "launch_base.h"

namespace muda
{
class stream
{
    enum class flag
    {
        block    = cudaStreamDefault,
        nonblock = cudaStreamNonBlocking
    };
    cudaStream_t handle;

  public:
    [[nodiscard]] stream(flag f = flag::nonblock)
    {
        checkCudaErrors(cudaStreamCreateWithFlags(&handle, (int)f));
    }
    ~stream() { checkCudaErrors(cudaStreamDestroy(handle)); }
    static sptr<stream> create(flag f = flag::nonblock)
    {
        return std::make_shared<stream>(f);
    }
    operator cudaStream_t() { return handle; }
};
}  // namespace muda