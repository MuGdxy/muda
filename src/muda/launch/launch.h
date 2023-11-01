#pragma once
#include <muda/launch/launch_base.h>

namespace muda
{
namespace details
{
    template <typename F, typename UserTag = DefaultTag>
    MUDA_GLOBAL void generic_kernel(F f)
    {
        f();
    }
}  // namespace details

// using details::generic_kernel;

class Launch : public LaunchBase<Launch>
{
    dim3   m_grid_dim;
    dim3   m_block_dim;
    size_t m_shared_mem_size;

  public:
    MUDA_HOST Launch(dim3 gridDim, dim3 blockDim, size_t sharedMemSize = 0, cudaStream_t stream = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_grid_dim(gridDim),
          m_block_dim(blockDim),
          m_shared_mem_size(sharedMemSize)
    {
    }

    MUDA_HOST Launch(int          gridDim       = 1,
                     int          blockDim      = 1,
                     size_t       sharedMemSize = 0,
                     cudaStream_t stream        = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_grid_dim(gridDim),
          m_block_dim(blockDim),
          m_shared_mem_size(sharedMemSize)
    {
    }

    template <typename F, typename UserTag = DefaultTag>
    MUDA_HOST Launch& apply(F&& f, UserTag tag = {});

    template <typename F, typename UserTag = DefaultTag>
    MUDA_HOST MUDA_NODISCARD S<KernelNodeParms<raw_type_t<F>>> as_node_parms(F&& f, UserTag tag = {});

  private:
    template <typename F, typename UserTag = DefaultTag>
    MUDA_HOST void invoke(F&& f, UserTag tag = {});
};
}  // namespace muda

#include "details/launch.inl"