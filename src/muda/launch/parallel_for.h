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
        KernelData(int count, U&& callable)
            : count(count)
            , callable(std::forward<U>(callable))
        {
        }
    };

    /// <summary>
    /// calculate grid dim automatically to cover the range
    /// </summary>
    /// <param name="blockDim">block dim to use</param>
    /// <param name="sharedMemSize"></param>
    /// <param name="stream"></param>
    ParallelFor(int blockDim, size_t shared_mem_size = 0, cudaStream_t stream = nullptr)
        : LaunchBase(stream)
        , m_gridDim(0)
        , m_block_dim(blockDim)
        , m_shared_mem_size(shared_mem_size)
    {
    }

    /// <summary>
    /// use Grid-Stride Loops to cover the range
    /// </summary>
    /// <param name="blockDim"></param>
    /// <param name="gridDim"></param>
    /// <param name="sharedMemSize"></param>
    /// <param name="stream"></param>
    ParallelFor(int gridDim, int blockDim, size_t shared_mem_size = 0, cudaStream_t stream = nullptr)
        : LaunchBase(stream)
        , m_gridDim(gridDim)
        , m_block_dim(blockDim)
        , m_shared_mem_size(shared_mem_size)
    {
    }

    template <typename F, typename UserTag = DefaultTag>
    ParallelFor& apply(int count, F&& f, UserTag tag = {})
    {
        using CallableType = raw_type_t<F>;
        static_assert(std::is_invocable_v<CallableType, int>, "f:void (int i)");

        check_input(count);

        if(m_gridDim <= 0)  // ParallelFor
        {
            if(count > 0)
            {
                // calculate the blocks we need
                auto n_blocks = calculate_grid_dim(count);
                details::parallel_for_kernel<CallableType, UserTag>
                    <<<n_blocks, m_block_dim, m_shared_mem_size, m_stream>>>(f, count);
            }
        }
        else  // grid stride loop
        {
            details::grid_stride_loop_kernel<CallableType, UserTag>
                <<<m_gridDim, m_block_dim, m_shared_mem_size, m_stream>>>(f, count);
        }
        return finish_kernel_launch();
    }

    template <typename F, typename UserTag = DefaultTag>
    MUDA_NODISCARD auto as_node_parms(int count, F&& f, UserTag tag = {})
    {
        using CallableType = raw_type_t<F>;
        static_assert(std::is_invocable_v<CallableType, int>, "f:void (int i)");

        check_input(count);

        auto parms = std::make_shared<KernelNodeParms<KernelData<CallableType>>>(
            count, std::forward<F>(f));
        if(m_gridDim <= 0)  // dynamic grid dim
        {
            auto n_blocks = calculate_grid_dim(count);
            parms->func((void*)details::parallel_for_kernel<CallableType, UserTag>);
            parms->gridDim(n_blocks);
        }
        else  // grid-stride loop
        {
            parms->func((void*)details::grid_stride_loop_kernel<CallableType, UserTag>);
            parms->gridDim(m_gridDim);
        }

        parms->blockDim(m_block_dim);
        parms->sharedMemBytes(m_shared_mem_size);
        parms->parse(
            [](KernelData<CallableType>& p) -> std::vector<void*> {
                return {&p.callable, &p.count};
            });
        finish_kernel_launch();
        return parms;
    }

  private:
    int calculate_grid_dim(int count)
    {
        auto nMinthread = count;
        auto nMinblocks = (nMinthread + m_block_dim - 1) / m_block_dim;
        return nMinblocks;
    }

    void check_input(int count)
    {
        if(count < 0)
            throw std::logic_error("count must be >= 0");
        if(m_block_dim <= 0)
            throw std::logic_error("blockDim must be > 0");
    }
};
}  // namespace muda