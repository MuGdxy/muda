#pragma once
#include <muda/tools/flag.h>
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
class Event
{
    cudaEvent_t m_handle;

  public:
    enum class bit : unsigned int
    {
        eDefault = cudaEventDefault,           /**< Default event flag */
        eBlockingSync = cudaEventBlockingSync, /**< Event uses blocking synchronization */
        eDisableTiming = cudaEventDisableTiming, /**< Event will not record timing data */
        eInterprocess = cudaEventInterprocess /**< Event is suitable for interprocess use. cudaEventDisableTiming must be set */
    };

    MUDA_NODISCARD Event(flags<bit> flag = bit::eDisableTiming)
    {
        checkCudaErrors(cudaEventCreateWithFlags(&m_handle, static_cast<unsigned int>(flag)));
    }

    // elapsed time (in ms) between two events
    static float elapsed_time(cudaEvent_t start, cudaEvent_t stop)
    {
        float time;
        checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
        return time;
    }

    ~Event() { checkCudaErrors(cudaEventDestroy(m_handle)); }

    operator cudaEvent_t() { return m_handle; }

    // delete copy constructor and assignment operator
    Event(const Event&)            = delete;
    Event& operator=(const Event&) = delete;
    // allow move constructor and assignment operator
    Event(Event&&)            = default;
    Event& operator=(Event&&) = default;
};
}  // namespace muda