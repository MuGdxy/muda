#pragma once

namespace muda::details::buffer
{
template <typename BufferView>
class BufferInfoAccessor
{
  public:
    static auto cuda_pitched_ptr(const BufferView& b)
    {
        return b.cuda_pitched_ptr();
    }
};
}  // namespace muda::details::buffer
