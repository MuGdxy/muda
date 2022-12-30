#pragma once

#include "base.h"


namespace muda
{
class DeviceRadixSort : public AlgBase<DeviceRadixSort>
{
  public:
    DeviceRadixSort(cudaStream_t stream = nullptr)
        : AlgBase(stream)
    {
    }

    template <typename KeyT, typename ValueT>
    DeviceRadixSort& SortPairs(device_buffer<std::byte>& external_buffer,
                               KeyT*                     d_keys_out,
                               ValueT*                   d_values_out,
                               KeyT*                     d_keys_in,
                               ValueT*                   d_values_in,
                               int                       num_items,
                               int                       begin_bit = 0,
                               int end_bit = sizeof(KeyT) * 8);

    template <typename KeyT>
    DeviceRadixSort& SortKeys(device_buffer<std::byte>& external_buffer,
                              KeyT*                     d_keys_out,
                              KeyT*                     d_keys_in,
                              int                       num_items,
                              int                       begin_bit = 0,
                              int end_bit = sizeof(KeyT) * 8);
};
}  // namespace muda

#ifndef __INTELLISENSE__
#include "radix_sort.inl"
#endif