#pragma once
#include "launch_base.h"

namespace muda
{
enum class host_type
{
    host_cuda = 0, // host function called by cuda stream (async with the host)
    host_sync = 1 // host function called in place, which is sync with the host thread.
};

namespace details
{
    template <typename F>
    __host__ void __stdcall genericHostCall(void* userdata)
    {
        auto f = reinterpret_cast<F*>(userdata);
        (*f)();
    }

    template <typename F>
    __host__ void __stdcall deleteFunctionObject(void* userdata)
    {
        auto f = reinterpret_cast<F*>(userdata);
        delete f;
    }

    template <typename F>
    struct structured_for
    {
        template <typename U>
        __host__ structured_for(int begin, int end, int step, U&& f)
            : begin(begin)
            , end(end)
            , step(step)
            , f(std::forward<U>(f))
        {
        }

        int begin;
        int end;
        int step;
        F   f;
    };

    template <typename F>
    __host__ void __stdcall genericHostFor(void* userdata)
    {
        auto f     = reinterpret_cast<structured_for<F>*>(userdata);
        auto begin = f->begin;
        auto end   = f->end;
        auto step  = f->step;
        for(int i = begin; i < end; i += step)
        {
            f->f(i);
        }
    }
}  // namespace details


class host_call : public launch_base<host_call>
{
  public:
    host_call(host_type host = host_type::host_cuda, cudaStream_t stream = nullptr)
        : launch_base(stream)
        , ht(host){};

    template <typename F>
    host_call& apply(F&& f)
    {
        using CallableType = raw_type_t<F>;
        static_assert(std::is_invocable_v<CallableType>, "f:void (void)");
        auto userdata = new CallableType(std::forward<F>(f));
        if(ht == host_type::host_cuda)
        {
            checkCudaErrors(cudaLaunchHostFunc(
                stream_, details::genericHostCall<CallableType>, userdata));
            checkCudaErrors(cudaLaunchHostFunc(
                stream_, details::deleteFunctionObject<CallableType>, userdata));
        }
        else
        {
            details::genericHostCall<CallableType>(userdata);
            delete userdata;
        }
        return *this;
    }

    template <typename F>
    [[nodiscard]] auto asNodeParms(F&& f)
    {
        if(ht != host_type::host_cuda)
            throw std::logic_error("only <cuda host> can call asNodeParms");
        using CallableType = std::remove_all_extents_t<F>;
        auto parms = std::make_shared<hostNodeParms<CallableType>>(std::forward<F>(f));
        parms->fn((cudaHostFn_t)details::genericHostCall<CallableType>);
        return parms;
    }

  private:
    host_type ht;
};

class host_for : public launch_base<host_for>
{
  public:
    host_for(host_type host = host_type::host_cuda, cudaStream_t stream = nullptr)
        : launch_base(stream)
        , ht(host){};


    template <typename F>
    host_for& apply(int begin, int end, int step, F&& f)
    {
        checkInput(begin, end, step);
        using CallableType = raw_type_t<F>;
        static_assert(std::is_invocable_v<CallableType, int>, "f:void (int i)");
        using comp_type = details::structured_for<CallableType>;
        auto sf         = new comp_type(begin, end, step, std::forward<F>(f));
        if(ht == host_type::host_cuda)
        {
            checkCudaErrors(
                cudaLaunchHostFunc(stream_, details::genericHostFor<CallableType>, sf));
            checkCudaErrors(cudaLaunchHostFunc(
                stream_, details::deleteFunctionObject<comp_type>, sf));
        }
        else
        {
            details::genericHostFor<CallableType>(sf);
            delete sf;
        }
        return *this;
    }

    template <typename F>
    host_for& apply(int begin, int count, F&& f)
    {
        return apply(begin, begin + count, 1, std::forward<F>(f));
    }

    template <typename F>
    host_for& apply(int count, F&& f)
    {
        return apply(0, count, 1, std::forward<F>(f));
    }

    template <typename F>
    [[nodiscard]] auto asNodeParms(int begin, int end, int step, F&& f)
    {
        if(ht != host_type::host_cuda)
            throw std::logic_error("only <cuda host> can call asNodeParms");
        checkInput(begin, end, step);
        using CallableType = std::remove_all_extents_t<F>;
        static_assert(std::is_invocable_v<CallableType, int>, "f:void (int i)");
        auto parms = std::make_shared<hostNodeParms<details::structured_for<CallableType>>>(
            details::structured_for<CallableType>(begin, end, step, std::forward<F>(f)));
        parms->fn((cudaHostFn_t)details::genericHostFor<CallableType>);
        return parms;
    }

    template <typename F>
    [[nodiscard]] auto asNodeParms(int begin, int count, F&& f)
    {
        return asNodeParms(begin, begin + count, 1, std::forward<F>(f));
    }

    template <typename F>
    [[nodiscard]] auto asNodeParms(int count, F&& f)
    {
        return asNodeParms(0, count, 1, std::forward<F>(f));
    }

  private:
    host_type   ht;
    static void checkInput(int begin, int end, int step)
    {
        if(step == 0)
            throw std::logic_error("step should not be 0!");
        else if(step * (end - begin) < 0)
            throw std::logic_error("step direction is not consistent with [begin, end)!");
        //else if(begin == end)
        //    throw std::logic_error("begin should not equal to end!");
    }
};
}  // namespace muda