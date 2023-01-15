#pragma once
#include "launch_base.h"

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

    [[nodiscard]] static sptr<stream> create(flag f = flag::block)
    {
        return std::make_shared<stream>(f);
    }
    
    operator cudaStream_t() { return m_handle; }
};
}  // namespace muda