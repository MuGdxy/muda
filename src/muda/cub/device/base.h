#pragma once
#include <muda/buffer.h>
#include <muda/container.h>
#include <muda/launch/launch_base.h>

namespace muda
{
template <typename Derive>
class CubWrapper : public LaunchBase<Derive>
{
  protected:
    void prepareBuffer(DeviceVector<std::byte>& buf, size_t reqSize)
    {
        // details::set_stream_check(buf, this->stream());
        buf.resize(reqSize);
    }

    void prepareBuffer(DeviceBuffer<std::byte>& buf, size_t reqSize)
    {
        details::set_stream_check(buf, this->stream());
        buf.resize(reqSize);
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
    prepareBuffer(external_buffer, temp_storage_bytes);                        \
    d_temp_storage = (void*)external_buffer.data();                            \
                                                                               \
    checkCudaErrors(x);                                                        \
                                                                               \
    return *this;
