#pragma once
#include <muda/buffer.h>
#include <muda/launch/launch_base.h>

namespace muda
{
template <typename Derive>
class CubWrapper : public launch_base<Derive>
{
  protected:
    void prepareBuffer(device_buffer<std::byte>& buf, size_t reqSize)
    {
        details::set_stream_check(buf, this->m_stream);
        buf.resize(reqSize);
    }

  public:
    CubWrapper(cudaStream_t stream = nullptr)
        : launch_base<Derive>(stream)
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
