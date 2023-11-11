#pragma once
#include <muda/launch/launch_base.h>
#include <muda/buffer.h>
#include <muda/container.h>
#include <muda/buffer/buffer_launch.h>
#include <muda/compute_graph/compute_graph.h>

namespace muda
{
template <typename Derive>
class CubWrapper : public LaunchBase<Derive>
{
  protected:
    void prepare_buffer(DeviceVector<std::byte>& buf, size_t reqSize)
    {
        // details::set_stream_check(buf, this->stream());
        buf.resize(reqSize);
    }

    void prepare_buffer(DeviceBuffer<std::byte>& buf, size_t reqSize)
    {
        BufferLaunch(m_stream).resize(buf, reqSize);
    }

  public:
    CubWrapper(cudaStream_t stream = nullptr)
        : LaunchBase<Derive>(stream)
    {
    }

    // meaningless for cub, so we just delete it
    void kernel_name(std::string_view) = delete;
};
}  // namespace muda
