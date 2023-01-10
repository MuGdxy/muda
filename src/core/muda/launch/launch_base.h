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

namespace muda
{
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
  protected:
    cudaStream_t m_stream;

  public:
    launch_base(cudaStream_t stream)
        : m_stream(stream)
    {
    }
    void push_range(const std::string& name)
    {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version               = NVTX_VERSION;
        eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType             = NVTX_COLOR_ARGB;
        eventAttrib.color                 = 255;
        eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii         = name.c_str();
        nvtxRangePushEx(&eventAttrib);
    }

    void pop_range() { nvtxRangePop(); }

    template <typename T>
    friend class launch_base;
    Derived& wait()
    {
        checkCudaErrors(cudaStreamSynchronize(m_stream));

        return derived();
    }

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
        n.m_stream = m_stream;
        return n;
    }

    template <typename Next, typename ...Args>
    Next next(Args&& ... args)
    {
        static_assert(std::is_base_of_v<launch_base<Next>, Next>, "not supported");
        Next n(std::forward<Args>(args)...);
        n.m_stream = m_stream;
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