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
    cudaStream_t m_handle;

  public:
    enum class flag : unsigned int
    {
        eDefault     = cudaStreamDefault,
        eNonBlocking = cudaStreamNonBlocking
    };

    MUDA_NODISCARD stream(flag f = flag::eDefault)
    {
        checkCudaErrors(cudaStreamCreateWithFlags(&m_handle, static_cast<unsigned int>(f)));
    }

    ~stream() { checkCudaErrors(cudaStreamDestroy(m_handle)); }

    operator cudaStream_t() { return m_handle; }
};
}  // namespace muda