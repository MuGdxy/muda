#pragma once
#include <muda/tools/flag.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "../check/check_cuda_errors.h"

namespace muda
{
/// <summary>
/// RAII wrapper for cudaEvent
/// </summary>
class Event
{
    cudaEvent_t m_handle;

  public:
    enum class Bit : unsigned int
    {
        eDefault = cudaEventDefault,           /**< Default event flag */
        eBlockingSync = cudaEventBlockingSync, /**< Event uses blocking synchronization */
        eDisableTiming = cudaEventDisableTiming, /**< Event will not record timing data */
        eInterprocess = cudaEventInterprocess /**< Event is suitable for interprocess use. cudaEventDisableTiming must be set */
    };

    enum class QueryResult
    {
        eFinished = cudaSuccess,       /**< The event has been recorded */
        eNotReady = cudaErrorNotReady, /**< The event has not been recorded yet */
    };

    Event(Flags<Bit> flag = Bit::eDisableTiming)
    {
        checkCudaErrors(cudaEventCreateWithFlags(&m_handle, static_cast<unsigned int>(flag)));
    }

    auto query() const
    {
        auto res = cudaEventQuery(m_handle);
        if(res != cudaSuccess && res != cudaErrorNotReady)
            checkCudaErrors(res);
        return static_cast<QueryResult>(res);
    }

    // elapsed time (in ms) between two events
    static auto elapsed_time(cudaEvent_t start, cudaEvent_t stop)
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