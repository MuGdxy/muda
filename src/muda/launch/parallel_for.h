#pragma once
#include "launch_base.h"
#include <stdexcept>
#include <exception>

namespace muda
{
namespace details
{
    /*
    **************************************************************************
    * This part is the core of the "launch part of muda"                     *
    **************************************************************************
    * F: the callable object                                                 *
    * UserTag: the tag struct for user to recognize on profiling             *
    **************************************************************************
    */

    template <typename F, typename UserTag>
    MUDA_GLOBAL void parallel_for_kernel(F f, int count)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto i   = tid;
        if(i < count)
            f(i);
    }

    /// <summary>
    ///
    /// </summary>
    template <typename F, typename UserTag>
    MUDA_GLOBAL void grid_stride_loop_kernel(F f, int count)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto i   = tid;
        for(; i < count; i += blockDim.x * gridDim.x)
            f(i);
    }
}  // namespace details


/// <summary>
/// ParallelFor
/// usage:
///		ParallelFor(16)
///			.apply(16, [=] __device__(int i) mutable { printf("var=%d, i = %d\n");}, true);
/// </summary>
class ParallelFor : public LaunchBase<ParallelFor>
{
    int    m_gridDim;
    int    m_block_dim;
    size_t m_shared_mem_size;

  public:
    template <typename F>
    class KernelData
    {
      public:
        int count;
        F   callable;
        template <typename U>
        KernelData(int count, U&& callable) MUDA_NOEXCEPT
            : count(count),
              callable(std::forward<U>(callable))
        {
        }
    };

    /// <summary>
    /// calculate grid dim automatically to cover the range
    /// </summary>
    /// <param name="blockDim">block dim to use</param>
    /// <param name="sharedMemSize"></param>
    /// <param name="stream"></param>
    ParallelFor(int blockDim, size_t shared_mem_size = 0, cudaStream_t stream = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_gridDim(0),
          m_block_dim(blockDim),
          m_shared_mem_size(shared_mem_size)
    {
    }

    /// <summary>
    /// use Grid-Stride Loops to cover the range
    /// </summary>
    /// <param name="blockDim"></param>
    /// <param name="gridDim"></param>
    /// <param name="sharedMemSize"></param>
    /// <param name="stream"></param>
    ParallelFor(int gridDim, int blockDim, size_t shared_mem_size = 0, cudaStream_t stream = nullptr) MUDA_NOEXCEPT
        : LaunchBase(stream),
          m_gridDim(gridDim),
          m_block_dim(blockDim),
          m_shared_mem_size(shared_mem_size)
    {
    }

    template <typename F, typename UserTag = DefaultTag>
    ParallelFor& apply(int count, F&& f, UserTag tag = {});

    template <typename F, typename UserTag = DefaultTag>
    MUDA_NODISCARD auto as_node_parms(int count, F&& f, UserTag tag = {})
        -> S<KernelNodeParms<KernelData<raw_type_t<F>>>>;

  private:
    template <typename F, typename UserTag = DefaultTag>
    void invoke(int count, F&& f, UserTag tag = {});

    int calculate_grid_dim(int count) const MUDA_NOEXCEPT;

    void check_input(int count) const MUDA_NOEXCEPT;
};
}  // namespace muda

#include <muda/launch/details/parallel_for.inl>