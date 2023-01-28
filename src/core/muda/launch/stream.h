#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "../check/checkCudaErrors.h"

namespace muda
{
/// <summary>
/// RAII wrapper for cudaStream
/// </summary>
class stream
{
    enum class flag
    {
        block    = cudaStreamDefault,
        nonblock = cudaStreamNonBlocking
    };
    cudaStream_t m_handle;

  public:
    [[nodiscard]] stream(flag f = flag::block)
    {
        checkCudaErrors(cudaStreamCreateWithFlags(&m_handle, (int)f));
    }

    ~stream() { checkCudaErrors(cudaStreamDestroy(m_handle)); }
    
    operator cudaStream_t() { return m_handle; }
};
}  // namespace muda