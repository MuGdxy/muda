#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "../check/check_cuda_errors.h"

namespace muda
{
/// <summary>
/// RAII wrapper for cudaStream
/// </summary>
class Stream
{
    cudaStream_t m_handle;

  public:
    enum class flag : unsigned int
    {
        eDefault     = cudaStreamDefault,
        eNonBlocking = cudaStreamNonBlocking
    };

    MUDA_NODISCARD Stream(flag f = flag::eDefault)
    {
        checkCudaErrors(cudaStreamCreateWithFlags(&m_handle, static_cast<unsigned int>(f)));
    }

    ~Stream() { checkCudaErrors(cudaStreamDestroy(m_handle)); }

    operator cudaStream_t() const { return m_handle; }

    // delete copy constructor and copy assignment operator
    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    // allow move constructor and move assignment operator
    Stream(Stream&&)            = default;
    Stream& operator=(Stream&&) = default;

    void wait() const { checkCudaErrors(cudaStreamSynchronize(m_handle)); }

    void begin_capture(cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal) const
    {
        checkCudaErrors(cudaStreamBeginCapture(m_handle, mode));
    }

    void end_capture(cudaGraph_t* graph) const
    {
        checkCudaErrors(cudaStreamEndCapture(m_handle, graph));
    }
};
}  // namespace muda