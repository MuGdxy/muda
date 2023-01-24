#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "../check/checkCudaErrors.h"

namespace muda
{
/// <summary>
/// RAII wrapper for cudaEvent
/// </summary>
class event
{
    cudaEvent_t m_handle;
  public:
    // flags:
    // cudaEventDefault         /**< Default event flag */
    // cudaEventBlockingSync    /**< Event uses blocking synchronization */
    // cudaEventDisableTiming   /**< Event will not record timing data */
    // cudaEventInterprocess    /**< Event is suitable for interprocess use. cudaEventDisableTiming must be set */
    [[nodiscard]] event(int flag = cudaEventDefault)
    {
        checkCudaErrors(cudaEventCreateWithFlags(&m_handle, flag));
    }
    
    // elapsed time (in ms) between two events
    static float elapsed_time(cudaEvent_t start, cudaEvent_t stop) {
        float time;
        checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
        return time;
    }

    ~event() { checkCudaErrors(cudaEventDestroy(m_handle)); }

    operator cudaEvent_t() { return m_handle; }
};
}  // namespace muda