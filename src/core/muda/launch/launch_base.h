#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <string>
#include <functional>
#include <memory>
#include <cooperative_groups.h>

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>
#include "../check/checkCudaErrors.h"
#include "../type_traits/type_mod.h"
#include "../graph.h"
#include "event.h"

namespace muda
{
/// <summary>
/// a default tag for kernel (which can be shown in profile tools such as Nsight System)
/// </summary>
struct DefaultTag
{
};
namespace details
{	
    inline void streamErrorCallback(cudaStream_t stream, cudaError error, void* userdata)
    {
        auto callback =
            reinterpret_cast<std::function<void(cudaStream_t, cudaError)>*>(userdata);
        (*callback)(stream, error);
        delete callback;
    }
}  // namespace details


template <typename Derived>
class launch_base
{
	template<typename Others>
    friend class launch_base;
	
  protected:
    cudaStream_t m_stream;
  public:
    launch_base(cudaStream_t stream)
        : m_stream(stream)
    {
    }

    virtual void init_stream(cudaStream_t s) { m_stream = s; }
	
    // create a named scope for better recognization (if you are using some profile tools)
    // usage:
    //  on(stream)
    //      .push_range("part1")
    //      .next<launch>(1,1).apply(...)
    //      .pop_range()
    //      .wait();
    Derived& push_range(const std::string& name)
    {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version               = NVTX_VERSION;
        eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType             = NVTX_COLOR_ARGB;
        eventAttrib.color                 = 255;
        eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii         = name.c_str();
        nvtxRangePushEx(&eventAttrib);
        return derived();
    }

    Derived& pop_range()
    {
        nvtxRangePop();
        return derived();
    }

    template <typename T>
    friend class launch_base;

    // record an event on this point with current stream, you could use .when() to
    // capture this event for synchronization
    // flags:
    //  cudaEventRecordDefault : Default event creation flag.
    //  cudaEventRecordExternal : Event is captured in the graph as an external
    Derived& record(cudaEvent_t e, int flag = cudaEventRecordDefault)
    {
        checkCudaErrors(cudaEventRecordWithFlags(e, m_stream, flag));

        return derived();
    }

    // let the following kernels wait until the event is triggered
    // (asynchronize with the host)
    // usage:
    //  on(stream)
    //      .when(event)
    //      .next<launch>(1,1).apply(...)
    //      .wait();
    // flags:
    //  cudaEventRecordDefault : Default event creation flag.
    //  cudaEventRecordExternal : Event is captured in the graph as an external
    Derived& when(cudaEvent_t e, int flag = cudaEventRecordDefault)
    {
        checkCudaErrors(cudaStreamWaitEvent(m_stream, e, flag));

        return derived();
    }

    // let the host wait for the event
    Derived& wait(cudaEvent_t e)
    {
        checkCudaErrors(cudaEventSynchronize(e));

        return derived();
    }

    // let the host wait for the current stream
    Derived& wait()
    {
        checkCudaErrors(cudaStreamSynchronize(m_stream));

        return derived();
    }

    // register a host callback function, which will be called when all the jobs before
    // this point are done.
    Derived& callback(const std::function<void(cudaStream_t, cudaError)>& callback)
    {
        auto userdata = new std::function<void(cudaStream_t, cudaError)>(callback);
        checkCudaErrors(
            cudaStreamAddCallback(m_stream, details::streamErrorCallback, userdata, 0));
        return derived();
    }

    template <typename Next>
    Next next(Next n)
    {
        static_assert(std::is_base_of_v<launch_base<Next>, Next>, "not supported");
        n.init_stream(m_stream);
        return n;
    }

    template <typename Next, typename... Args>
    Next next(Args&&... args)
    {
        static_assert(std::is_base_of_v<launch_base<Next>, Next>, "not supported");
        Next n(std::forward<Args>(args)...);
        n.init_stream(m_stream);
        return n;
    }

  private:
    Derived& derived() { return *(Derived*)(this); }
};

class empty : public launch_base<empty>
{
  public:
    empty(cudaStream_t stream)
        : launch_base(stream)
    {
    }
};

inline empty on(cudaStream_t stream = nullptr)
{
    return empty(stream);
}
}  // namespace muda