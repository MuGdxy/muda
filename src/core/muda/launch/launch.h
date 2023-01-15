#pragma once
#include "launch_base.h"

namespace muda
{
namespace details
{
    template <typename F>
    __global__ void genericKernel(F f)
    {
        f();
    }
}  // namespace details

class launch : public launch_base<launch>
{
    dim3   m_gridDim;
    dim3   m_blockDim;
    size_t m_sharedMemSize;

  public:
    launch(dim3 gridDim, dim3 blockDim, size_t sharedMemSize = 0, cudaStream_t stream = nullptr)
        : launch_base(stream)
        , m_gridDim(gridDim)
        , m_blockDim(blockDim)
        , m_sharedMemSize(sharedMemSize)
    {
    }

    launch(int gridDim = 1, int blockDim = 1, size_t sharedMemSize = 0, cudaStream_t stream = nullptr)
        : launch_base(stream)
        , m_gridDim(gridDim)
        , m_blockDim(blockDim)
        , m_sharedMemSize(sharedMemSize)
    {
    }

    template <typename F>
    launch& apply(F&& f)
    {
        details::genericKernel<<<m_gridDim, m_blockDim, m_sharedMemSize, m_stream>>>(f);
        return *this;
    }

    template <typename F>
    [[nodiscard]] auto asNodeParms(F&& f)
    {
        using CallableType = raw_type_t<F>;
        static_assert(std::is_invocable_v<CallableType>, "f:void (void)");
        auto parms =
            std::make_shared<kernelNodeParms<CallableType>>(std::forward<F>(f));

        parms->func((void*)details::genericKernel<CallableType>);
        parms->gridDim(m_gridDim);
        parms->blockDim(m_blockDim);
        parms->sharedMemBytes(m_sharedMemSize);
        parms->parse([](CallableType& p) -> std::vector<void*> { return {&p}; });
        return parms;
    }

    static void wait_stream(cudaStream_t stream)
    {
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
    static void wait_device() { checkCudaErrors(cudaDeviceSynchronize()); }
};
}  // namespace muda