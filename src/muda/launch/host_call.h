#pragma once
#include "launch_base.h"

namespace muda
{
namespace details
{
    template <typename F>
    __host__ void __stdcall genericHostCall(void* userdata)
    {
        auto f = reinterpret_cast<F*>(userdata);
        (*f)();
        delete f;
    }
}  // namespace details


class host_call : public launch_base<host_call>
{
  public:
    host_call(cudaStream_t stream = nullptr)
        : launch_base(stream){};

    template <typename F>
    host_call& apply(F&& f)
    {
        using CallableType = raw_type_t<F>;
        auto userdata      = new CallableType(std::forward<F>(f));
        checkCudaErrors(
            cudaLaunchHostFunc(stream_, details::genericHostCall<CallableType>, userdata));
        return *this;
    }

    template <typename F>
    [[nodiscard]] static auto asNodeParms(F&& f)
    {
        using CallableType = std::remove_all_extents_t<F>;
        auto userdata      = new CallableType(std::forward<F>(f));
        auto parms = std::make_shared<hostNodeParms<CallableType>>(userdata);
        parms->fn((cudaHostFn_t)details::genericHostCall<CallableType>);
        return parms;
    }
};
}  // namespace muda