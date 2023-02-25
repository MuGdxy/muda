#pragma once
#include "launch_base.h"

namespace muda
{
namespace details
{
    template <typename F, typename UserTag>
    MUDA_HOST void CUDARTAPI genericHostCall(void* userdata)
    {
        auto f = reinterpret_cast<F*>(userdata);
        (*f)();
    }

    template <typename F, typename UserTag>
    MUDA_HOST void CUDARTAPI deleteFunctionObject(void* userdata)
    {
        auto f = reinterpret_cast<F*>(userdata);
        delete f;
    }
}  // namespace details


class host_call : public launch_base<host_call>
{
  public:
    host_call(cudaStream_t stream = nullptr)
        : launch_base(stream)
    {
    }

    template <typename F, typename UserTag = DefaultTag>
    host_call& apply(F&& f, UserTag tag = {})
    {
        using CallableType = raw_type_t<F>;
        static_assert(std::is_invocable_v<CallableType>, "f:void (void)");
        auto userdata = new CallableType(std::forward<F>(f));
        checkCudaErrors(cudaLaunchHostFunc(
            m_stream, details::genericHostCall<CallableType, UserTag>, userdata));
        checkCudaErrors(cudaLaunchHostFunc(
            m_stream, details::deleteFunctionObject<CallableType, UserTag>, userdata));
        return *this;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="F"></typeparam>
    /// <typeparam name="UserTag"></typeparam>
    /// <param name="f"></param>
    /// <param name="tag"></param>
    /// <returns></returns>
    template <typename F, typename UserTag = DefaultTag>
    MUDA_NODISCARD auto asNodeParms(F&& f, UserTag tag = {})
    {
        using CallableType = raw_type_t<F>;
        auto parms = std::make_shared<hostNodeParms<CallableType>>(std::forward<F>(f));
        parms->fn((cudaHostFn_t)details::genericHostCall<CallableType, UserTag>);
        return parms;
    }
};
}  // namespace muda