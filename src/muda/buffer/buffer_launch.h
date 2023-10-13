#pragma once
#include <muda/launch/launch_base.h>
#include <muda/muda_config.h>
#include <muda/buffer/var_view.h>
#include <muda/buffer/buffer_view.h>

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
    BufferLaunch(cudaStream_t s = nullptr) MUDA_NOEXCEPT : LaunchBase(s) {}

    BufferLaunch(int block_dim, cudaStream_t s = nullptr) MUDA_NOEXCEPT
        : LaunchBase(s),
          m_block_dim(block_dim)
    {
    }

    BufferLaunch(int grid_dim, int block_dim, cudaStream_t s = nullptr) MUDA_NOEXCEPT
        : LaunchBase(s),
          m_grid_dim(grid_dim),
          m_block_dim(block_dim)
    {
    }

    template <typename T>
    BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size);
    template <typename T>
    BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size, const T& val);
    template <typename T>
    BufferLaunch& clear(DeviceBuffer<T>& buffer);
    template <typename T>
    BufferLaunch& alloc(DeviceBuffer<T>& buffer, size_t n);
    template <typename T>
    BufferLaunch& free(DeviceBuffer<T>& buffer);
    template <typename T>
    BufferLaunch& shrink_to_fit(DeviceBuffer<T>& buffer);

    template <typename T>
    BufferLaunch& copy(BufferView<T>& dst, const BufferView<T>& src);
    template <typename T>
    BufferLaunch& copy(T* dst, const BufferView<T>& src);
    template <typename T>
    BufferLaunch& copy(BufferView<T>& dst, const T* src);

    template <typename T>
    BufferLaunch& copy(VarView<T>& dst, const VarView<T>& src);
    template <typename T>
    BufferLaunch& copy(T* dst, const VarView<T>& src);
    template <typename T>
    BufferLaunch& copy(VarView<T>& dst, const T* src);

    template <typename T>
    BufferLaunch& fill(BufferView<T>& buffer, const T& val);
    template <typename T>
    BufferLaunch& fill(VarView<T>& buffer, const T& val);

  private:
    template <typename T, typename FConstruct>
    BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size, FConstruct&& fct);
};
}  // namespace muda

#include "details/buffer_launch.inl"