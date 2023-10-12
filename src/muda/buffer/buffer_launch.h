#include <muda/launch/launch_base.h>
#include <muda/muda_config.h>
namespace muda
{
template <typename T>
class DeviceBuffer;
class DeviceBufferVar;

class BufferLaunch : public LaunchBase<BufferLaunch>
{
    int m_grid_dim  = 0;
    int m_block_dim = LIGHT_WORKLOAD_BLOCK_SIZE;

  public:
    BufferLaunch(cudaStream_t s = nullptr);
    BufferLaunch(int block_dim, cudaStream_t s = nullptr);
    BufferLaunch(int grid_dim, int block_dim, cudaStream_t s = nullptr);

    template <typename T>
    BufferLaunch& resize(DeviceBuffer<T>& buffer, int size);

    template <typename T>
    BufferLaunch& resize(DeviceBuffer<T>& buffer, int size, const T& val);

  private:
    template <typename T, typename F>
    BufferLaunch& resize(DeviceBuffer<T>& buffer, int size, F&& f);
};
}  // namespace muda

#include "details/buffer_launch.inl"