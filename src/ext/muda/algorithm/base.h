#pragma once
#include <muda/buffer.h>
#include <muda/launch/launch_base.h>

namespace muda
{
template <typename Derive>
class AlgBase : public launch_base<Derive>
{
  protected:
    void prepareBuffer(device_buffer<std::byte>& buf, size_t reqSize) 
    {
        details::set_stream_check(buf, this->m_stream);
        buf.resize(reqSize);
    }
  public:
    AlgBase(cudaStream_t stream = nullptr)
        : launch_base<Derive>(stream)
    {
    }
};
}  // namespace muda::details