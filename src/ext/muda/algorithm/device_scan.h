#pragma once

#include "base.h"

namespace muda
{
class DeviceScan : public AlgBase<DeviceScan>
{
  public:
    DeviceScan(cudaStream_t stream = nullptr)
        : AlgBase(stream)
    {
    }

    template <typename T>
    DeviceScan& ExclusiveSum(device_buffer<std::byte>& external_buffer, T* d_out, T* d_in, int num_items);

    template <typename T>
    DeviceScan& ExclusiveSum(T* d_out, T* d_in, int num_items)
    {
        device_buffer<std::byte> external_buffer;
        return ExclusiveSum(external_buffer, d_out, d_in, num_items);
    }

    template <typename T>
    DeviceScan& InclusiveSum(device_buffer<std::byte>& external_buffer, T* d_out, T* d_in, int num_items);

    template <typename T>
    DeviceScan& InclusiveSum(T* d_out, T* d_in, int num_items)
    {
        device_buffer<std::byte> external_buffer;
        return InclusiveSum(external_buffer, d_out, d_in, num_items);
    }
};
}  // namespace muda

#ifndef __INTELLISENSE__
#include "./device_scan.inl"
#endif