#pragma once

#include "../launch/launch_base.h"
#include "../buffer.h"


namespace muda
{
class RadixSort : public launch_base<RadixSort>
{
  public:
    RadixSort(cudaStream_t stream = nullptr)
        : launch_base(stream)
    {
    }

    template <typename KeyT, typename ValueT>
    RadixSort& SortPairs(device_buffer<std::byte>& external_buffer,
                         KeyT*                     d_keys_out,
                         KeyT*                     d_keys_in,
                         ValueT*                   d_values_out,
                         ValueT*                   d_values_in,
                         int                       num_items,
                         int                       begin_bit = 0,
                         int                       end_bit = sizeof(KeyT) * 8);

    template <typename KeyT>
    RadixSort& SortKeys(device_buffer<std::byte>& external_buffer,
                        KeyT*                     d_keys_out,
                        KeyT*                     d_keys_in,
                        int                       num_items,
                        int                       begin_bit = 0,
                        int                       end_bit   = sizeof(KeyT) * 8);
};
}  // namespace muda

#ifndef __INTELLISENSE__
#include "radix_sort.inl"
#endif