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
};
}  // namespace muda

#define MUDA_CUB_WRAPPER_IMPL(x)                                               \
    size_t temp_storage_bytes = 0;                                             \
    void*  d_temp_storage     = nullptr;                                       \
                                                                               \
    checkCudaErrors(x);                                                        \
                                                                               \
    prepare_buffer(external_buffer, temp_storage_bytes);                        \
    d_temp_storage = (void*)external_buffer.data();                            \
                                                                               \
    checkCudaErrors(x);                                                        \
                                                                               \
    return *this;
