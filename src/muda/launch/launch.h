#pragma once
#include "launch_base.h"

namespace muda
{
namespace details
{
    template <typename F, typename UserTag>
    MUDA_GLOBAL void generic_kernel(F f)
    {
        f();
    }
}  // namespace details

class Launch : public LaunchBase<Launch>
{
    dim3   m_gridDim;
    dim3   m_block_dim;
    size_t m_shared_mem_size;

  public:
    Launch(dim3 gridDim, dim3 blockDim, size_t sharedMemSize = 0, cudaStream_t stream = nullptr)
        : LaunchBase(stream)
        , m_gridDim(gridDim)
        , m_block_dim(blockDim)
        , m_shared_mem_size(sharedMemSize)
    {
    }

    Launch(int gridDim = 1, int blockDim = 1, size_t sharedMemSize = 0, cudaStream_t stream = nullptr)
        : LaunchBase(stream)
        , m_gridDim(gridDim)
        , m_block_dim(blockDim)
        , m_shared_mem_size(sharedMemSize)
    {
    }

    template <typename F, typename UserTag = DefaultTag>
    Launch& apply(F&& f, UserTag tag = {})
    {
        using CallableType = raw_type_t<F>;
        static_assert(std::is_invocable_v<CallableType>, "f:void (void)");
        details::generic_kernel<CallableType, UserTag>
            <<<m_gridDim, m_block_dim, m_shared_mem_size, m_stream>>>(f);
        return finish_kernel_launch();
    }

    template <typename F, typename UserTag = DefaultTag>
    MUDA_NODISCARD auto as_node_parms(F&& f, UserTag tag = {})
    {
        using CallableType = raw_type_t<F>;
        static_assert(std::is_invocable_v<CallableType>, "f:void (void)");
        auto parms =
            std::make_shared<KernelNodeParms<CallableType>>(std::forward<F>(f));

        parms->func((void*)details::generic_kernel<CallableType, UserTag>);
        parms->gridDim(m_gridDim);
        parms->blockDim(m_block_dim);
        parms->sharedMemBytes(m_shared_mem_size);
        parms->parse([](CallableType& p) -> std::vector<void*> { return {&p}; });
        finish_kernel_launch();
        return parms;
    }

    static void wait_event(cudaEvent_t event)
    {
        checkCudaErrors(cudaEventSynchronize(event));
    }

    static void wait_stream(cudaStream_t stream)
    {
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    static void wait_device() { checkCudaErrors(cudaDeviceSynchronize()); }
};
}  // namespace muda