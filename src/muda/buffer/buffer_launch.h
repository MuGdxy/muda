#pragma once
#include <muda/launch/launch_base.h>
#include <muda/muda_config.h>
#include <muda/buffer/var_view.h>
#include <muda/buffer/buffer_view.h>
#include <muda/buffer/graph_buffer_view.h>
#include <muda/buffer/graph_var_view.h>

namespace muda
{
template <typename T>
class DeviceBuffer;

template <typename T>
class DeviceVar;

class BufferLaunch : public LaunchBase<BufferLaunch>
{
    int m_grid_dim  = 0;
    int m_block_dim = LIGHT_WORKLOAD_BLOCK_SIZE;

  public:
    MUDA_HOST BufferLaunch(cudaStream_t s = nullptr) MUDA_NOEXCEPT : LaunchBase(s)
    {
    }

    MUDA_HOST BufferLaunch(int block_dim, cudaStream_t s = nullptr) MUDA_NOEXCEPT
        : LaunchBase(s),
          m_block_dim(block_dim)
    {
    }

    MUDA_HOST BufferLaunch(int grid_dim, int block_dim, cudaStream_t s = nullptr) MUDA_NOEXCEPT
        : LaunchBase(s),
          m_grid_dim(grid_dim),
          m_block_dim(block_dim)
    {
    }

    template <typename T>
    MUDA_HOST BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size);
    template <typename T>
    MUDA_HOST BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size, const T& val);
    template <typename T>
    MUDA_HOST BufferLaunch& clear(DeviceBuffer<T>& buffer);
    template <typename T>
    MUDA_HOST BufferLaunch& alloc(DeviceBuffer<T>& buffer, size_t n);
    template <typename T>
    MUDA_HOST BufferLaunch& free(DeviceBuffer<T>& buffer);
    template <typename T>
    MUDA_HOST BufferLaunch& shrink_to_fit(DeviceBuffer<T>& buffer);

    // device to device
    template <typename T>
    MUDA_HOST BufferLaunch& copy(BufferView<T> dst, const BufferView<T>& src);
    template <typename T>
    MUDA_HOST BufferLaunch& copy(ComputeGraphVar<BufferView<T>>&       dst,
                                 const ComputeGraphVar<BufferView<T>>& src);

    // device to host
    template <typename T>
    MUDA_HOST BufferLaunch& copy(T* dst, const BufferView<T>& src);
    template <typename T>
    MUDA_HOST BufferLaunch& copy(ComputeGraphVar<T*>&                  dst,
                                 const ComputeGraphVar<BufferView<T>>& src);

    // host to device
    template <typename T>
    MUDA_HOST BufferLaunch& copy(BufferView<T> dst, const T* src);
    template <typename T>
    MUDA_HOST BufferLaunch& copy(ComputeGraphVar<BufferView<T>>& dst,
                                 const ComputeGraphVar<T*>&      src);

    // device to device
    template <typename T>
    MUDA_HOST BufferLaunch& copy(VarView<T> dst, const VarView<T>& src);
    template <typename T>
    MUDA_HOST BufferLaunch& copy(ComputeGraphVar<VarView<T>>&       dst,
                                 const ComputeGraphVar<VarView<T>>& src);

    // device to host
    template <typename T>
    MUDA_HOST BufferLaunch& copy(T* dst, const VarView<T>& src);
    template <typename T>
    MUDA_HOST BufferLaunch& copy(ComputeGraphVar<T*>&               dst,
                                 const ComputeGraphVar<VarView<T>>& src);

    // host to device
    template <typename T>
    MUDA_HOST BufferLaunch& copy(VarView<T> dst, const T* src);
    template <typename T>
    MUDA_HOST BufferLaunch& copy(ComputeGraphVar<VarView<T>>& dst,
                                 const ComputeGraphVar<T*>&   src);

    // host to device (scattered)
    template <typename T>
    MUDA_HOST BufferLaunch& fill(BufferView<T> buffer, const T& val);
    template <typename T>
    MUDA_HOST BufferLaunch& fill(ComputeGraphVar<BufferView<T>>& buffer,
                                 const ComputeGraphVar<T>&       val);

    // host to device (scattered)
    template <typename T>
    MUDA_HOST BufferLaunch& fill(VarView<T> buffer, const T& val);
    template <typename T>
    MUDA_HOST BufferLaunch& fill(ComputeGraphVar<VarView<T>>& buffer,
                                 const ComputeGraphVar<T>&    val);

  private:
    template <typename T, typename FConstruct>
    MUDA_HOST BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size, FConstruct&& fct);
};
}  // namespace muda

#include "details/buffer_launch.inl"