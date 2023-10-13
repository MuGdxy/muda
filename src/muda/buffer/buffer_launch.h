#pragma once
#include <muda/launch/launch_base.h>
#include <muda/muda_config.h>
#include <vector>

namespace muda
{
template <typename T>
class DeviceBuffer;

template <typename T>
class DeviceBufferVar;

class BufferLaunch : public LaunchBase<BufferLaunch>
{
    int m_grid_dim  = 0;
    int m_block_dim = LIGHT_WORKLOAD_BLOCK_SIZE;

  public:
    BufferLaunch(cudaStream_t s = nullptr)
        : LaunchBase(s)
    {
    }

    BufferLaunch(int block_dim, cudaStream_t s = nullptr)
        : LaunchBase(s)
        , m_block_dim(block_dim)
    {
    }

    BufferLaunch(int grid_dim, int block_dim, cudaStream_t s = nullptr)
        : LaunchBase(s)
        , m_grid_dim(grid_dim)
        , m_block_dim(block_dim)
    {
    }

    template <typename T>
    BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size);
    template <typename T>
    BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size, const T& val);
    template <typename T>
    BufferLaunch& clear(DeviceBuffer<T>& buffer);
    template <typename T>
    BufferLaunch& alloc(DeviceBuffer<T>& buffer);
    template <typename T>
    BufferLaunch& free(DeviceBuffer<T>& buffer);
    template <typename T>
    BufferLaunch& shrink_to_fit(DeviceBuffer<T>& buffer);

    template <typename T>
    BufferLaunch& copy(DeviceBufferView<T>& dst, const DeviceBufferView<T>& src);
    template <typename T>
    BufferLaunch& copy(T* dst, const DeviceBufferView<T>& src);
    template <typename T>
    BufferLaunch& copy(DeviceBufferView<T>& dst, const T* src);

    template <typename T>
    BufferLaunch& copy(DeviceBufferVar<T>& dst, const DeviceBufferVar<T>& src);
    template <typename T>
    BufferLaunch& copy(T* dst, const DeviceBufferVar<T>& src);
    template <typename T>
    BufferLaunch& copy(DeviceBufferVar<T>& dst, const T* src);
    template <typename T>
    BufferLaunch& copy(DeviceBufferVar<T>& dst, const DeviceBufferView<T>& src);
    template <typename T>
    BufferLaunch& copy(DeviceBufferView<T>& dst, const DeviceBufferVar<T>& src);

    template <typename T>
    BufferLaunch& fill(DeviceBufferView<T>& buffer, const T& val);
    template <typename T>
    BufferLaunch& fill(DeviceBufferVar<T>& buffer, const T& val);

  private:
    template <typename T, typename FConstruct>
    BufferLaunch& resize(DeviceBuffer<T>& buffer, size_t size, FConstruct&& fct);
};
}  // namespace muda

#include "details/buffer_launch.inl"