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
    Launch(dim3 gridDim, dim3 blockDim, size_t sharedMemSize = 0, cudaStream_t stream = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_gridDim(gridDim),
          m_block_dim(blockDim),
          m_shared_mem_size(sharedMemSize)
    {
    }

    Launch(int gridDim = 1, int blockDim = 1, size_t sharedMemSize = 0, cudaStream_t stream = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_gridDim(gridDim),
          m_block_dim(blockDim),
          m_shared_mem_size(sharedMemSize)
    {
    }

    template <typename F, typename UserTag = DefaultTag>
    Launch& apply(F&& f, UserTag tag = {});

    template <typename F, typename UserTag = DefaultTag>
    MUDA_NODISCARD S<KernelNodeParms<raw_type_t<F>>> as_node_parms(F&& f, UserTag tag = {});

    static void wait_event(cudaEvent_t event);

    static void wait_stream(cudaStream_t stream)
    {
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    static void wait_device() { checkCudaErrors(cudaDeviceSynchronize()); }

  private:
    template <typename F, typename UserTag = DefaultTag>
    void invoke(F&& f, UserTag tag = {});
};
}  // namespace muda

#include <muda/launch/details/launch.inl>