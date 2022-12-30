#pragma once
#include "base.h"

namespace muda
{
class DeviceReduce : public AlgBase<DeviceReduce>
{
  public:
    DeviceReduce(cudaStream_t stream = nullptr)
        : AlgBase(stream)
    {
    }

    template <typename T, typename Compare>
    DeviceReduce& Reduce(device_buffer<std::byte>& external_buffer,
                         T*                        d_out,
                         T*                        d_in,
                         int                       num_items,
                         Compare                   cmp,
                         T                         init);

    template <typename T>
    DeviceReduce& Max(device_buffer<std::byte>& external_buffer,
                      T*                        d_out,
                      T*                        d_in,
                      int                       num_items);
};

struct CustomMin
{
    template <typename T>
    __device__ __forceinline__ T operator()(const T& a, const T& b) const
    {
        return (b < a) ? b : a;
    }
};

}  // namespace muda

#ifndef __INTELLISENSE__
#include "reduce.inl"
#endif