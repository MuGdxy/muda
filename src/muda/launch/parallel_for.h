#pragma once
#include "launch_base.h"
#include <stdexcept>
#include <exception>

namespace muda
{
namespace details
{
    template <typename F, typename UserTag>
    MUDA_GLOBAL void parallel_for_kernel_with_details(F f, int count);

    template <typename F, typename UserTag>
    MUDA_GLOBAL void grid_stride_loop_kernel_with_details(F f, int count);
}  // namespace details

enum class ParallelForType : uint32_t
{
    DynamicBlocks,
    GridStrideLoop
};

class ParallelForDetails
{
  public:
    MUDA_NODISCARD MUDA_DEVICE int  active_num_in_block() const MUDA_NOEXCEPT;
    MUDA_NODISCARD MUDA_DEVICE bool is_final_block() const MUDA_NOEXCEPT;
    MUDA_NODISCARD MUDA_DEVICE auto parallel_for_type() const MUDA_NOEXCEPT
    {
        return m_type;
    }

    MUDA_NODISCARD MUDA_DEVICE int total_num() const MUDA_NOEXCEPT
    {
        return m_total_num;
    }
    MUDA_NODISCARD MUDA_DEVICE operator int() const MUDA_NOEXCEPT
    {
        return m_current_i;
    }

    MUDA_NODISCARD MUDA_DEVICE int i() const MUDA_NOEXCEPT
    {
        return m_current_i;
    }

    MUDA_NODISCARD MUDA_DEVICE int batch_i() const MUDA_NOEXCEPT
    {
        return m_batch_i;
    }

    MUDA_NODISCARD MUDA_DEVICE int total_batch() const MUDA_NOEXCEPT
    {
        return m_total_batch;
    }

  private:
    template <typename F, typename UserTag>
    friend MUDA_GLOBAL void details::parallel_for_kernel_with_details(F f, int count);

    template <typename F, typename UserTag>
    friend MUDA_GLOBAL void details::grid_stride_loop_kernel_with_details(F f, int count);

    MUDA_DEVICE ParallelForDetails(ParallelForType type, int i, int total_num) MUDA_NOEXCEPT
        : m_type(type),
          m_total_num(total_num),
          m_current_i(i)
    {
    }

    ParallelForType m_type;
    int             m_total_num;
    int             m_total_batch         = 1;
    int             m_batch_i             = 0;
    int             m_active_num_in_block = 0;
    int             m_current_i           = 0;
};

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
        {
            f(i);
        }
    }

    template <typename F, typename UserTag>
    MUDA_GLOBAL void parallel_for_kernel_with_details(F f, int count)
    {
        ParallelForDetails details{ParallelForType::DynamicBlocks,
                                   static_cast<int>(blockIdx.x * blockDim.x
                                                    + threadIdx.x),
                                   count};
        if(details.i() < count)
        {
            f(details);
        }
    }

    template <typename F, typename UserTag>
    MUDA_GLOBAL void grid_stride_loop_kernel(F f, int count)
    {
        auto tid       = blockIdx.x * blockDim.x + threadIdx.x;
        auto grid_size = gridDim.x * blockDim.x;
        auto i         = tid;
        for(; i < count; i += grid_size)
            f(i);
    }

    template <typename F, typename UserTag>
    MUDA_GLOBAL void grid_stride_loop_kernel_with_details(F f, int count)
    {
        auto tid        = blockIdx.x * blockDim.x + threadIdx.x;
        auto grid_size  = gridDim.x * blockDim.x;
        auto block_size = blockDim.x;
        auto i          = tid;
        auto round      = (count + grid_size - 1) / grid_size;
        for(int j = 0; i < count; i += grid_size, ++j)
        {
            ParallelForDetails details{ParallelForType::GridStrideLoop, static_cast<int>(i), count};
            details.m_total_batch = round;
            details.m_batch_i     = j;
            if(i + block_size > count)  // the block may be incomplete in the last round
                details.m_active_num_in_block = count - j * grid_size;
            else
                details.m_active_num_in_block = block_size;
            f(details);
        }
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

    static int round_up_blocks(int count, int block_dim) MUDA_NOEXCEPT
    {
        return (count + block_dim - 1) / block_dim;
    }

  private:
    template <typename F, typename UserTag = DefaultTag>
    void invoke(int count, F&& f, UserTag tag = {});

    int calculate_grid_dim(int count) const MUDA_NOEXCEPT;

    void check_input(int count) const MUDA_NOEXCEPT;
};
}  // namespace muda

#include <muda/launch/details/parallel_for.inl>