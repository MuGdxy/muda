#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_radix_sort.cuh>
#else
namespace cub
{
template <typename KeyT>
class DoubleBuffer;
}
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
class DeviceRadixSort : public CubWrapper<DeviceRadixSort>
{
  public:
    DeviceRadixSort(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    template <typename KeyT, typename ValueT, typename NumItemsT>
    DeviceRadixSort& SortPairs(DeviceBuffer<std::byte>& external_buffer,
                               const KeyT*               d_keys_in,
                               KeyT*                     d_keys_out,
                               const ValueT*             d_values_in,
                               ValueT*                   d_values_out,
                               NumItemsT                 num_items,
                               int                       begin_bit = 0,
                               int  end_bit           = sizeof(KeyT) * 8,
                               bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                                              temp_storage_bytes,
                                                              d_keys_in,
                                                              d_keys_out,
                                                              d_values_in,
                                                              d_values_out,
                                                              num_items,
                                                              begin_bit,
                                                              end_bit,
                                                              m_stream,
                                                              debug_synchronous));
    }

    template <typename KeyT, typename ValueT, typename NumItemsT>
    DeviceRadixSort& SortPairs(DeviceBuffer<std::byte>&  external_buffer,
                               cub::DoubleBuffer<KeyT>&   d_keys,
                               cub::DoubleBuffer<ValueT>& d_values,
                               NumItemsT                  num_items,
                               int                        begin_bit = 0,
                               int  end_bit           = sizeof(KeyT) * 8,
                               bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, m_stream, debug_synchronous));
    }

    template <typename KeyT, typename ValueT, typename NumItemsT>
    DeviceRadixSort& SortPairsDescending(DeviceBuffer<std::byte>& external_buffer,
                                         const KeyT*   d_keys_in,
                                         KeyT*         d_keys_out,
                                         const ValueT* d_values_in,
                                         ValueT*       d_values_out,
                                         NumItemsT     num_items,
                                         int           begin_bit = 0,
                                         int  end_bit = sizeof(KeyT) * 8,
                                         bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(
            cub::DeviceRadixSort::SortPairsDescending(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_keys_in,
                                                      d_keys_out,
                                                      d_values_in,
                                                      d_values_out,
                                                      num_items,
                                                      begin_bit,
                                                      end_bit,
                                                      m_stream,
                                                      debug_synchronous));
    }

    template <typename KeyT, typename ValueT, typename NumItemsT>
    DeviceRadixSort& SortPairsDescending(DeviceBuffer<std::byte>& external_buffer,
                                         cub::DoubleBuffer<KeyT>&   d_keys,
                                         cub::DoubleBuffer<ValueT>& d_values,
                                         NumItemsT                  num_items,
                                         int  begin_bit = 0,
                                         int  end_bit   = sizeof(KeyT) * 8,
                                         bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, m_stream, debug_synchronous));
    }

    template <typename KeyT, typename NumItemsT>
    DeviceRadixSort& SortKeys(DeviceBuffer<std::byte>& external_buffer,
                              const KeyT*               d_keys_in,
                              KeyT*                     d_keys_out,
                              NumItemsT                 num_items,
                              int                       begin_bit = 0,
                              int  end_bit           = sizeof(KeyT) * 8,
                              bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, m_stream, debug_synchronous));
    }

    template <typename KeyT, typename NumItemsT>
    DeviceRadixSort& SortKeys(DeviceBuffer<std::byte>& external_buffer,
                              cub::DoubleBuffer<KeyT>&  d_keys,
                              NumItemsT                 num_items,
                              int                       begin_bit = 0,
                              int  end_bit           = sizeof(KeyT) * 8,
                              bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, m_stream, debug_synchronous));
    }

    template <typename KeyT, typename NumItemsT>
    DeviceRadixSort& SortKeysDescending(DeviceBuffer<std::byte>& external_buffer,
                                        const KeyT* d_keys_in,
                                        KeyT*       d_keys_out,
                                        NumItemsT   num_items,
                                        int         begin_bit = 0,
                                        int         end_bit = sizeof(KeyT) * 8,
                                        bool        debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, m_stream, debug_synchronous));
    }

    template <typename KeyT, typename NumItemsT>
    DeviceRadixSort& SortKeysDescending(DeviceBuffer<std::byte>& external_buffer,
                                        cub::DoubleBuffer<KeyT>& d_keys,
                                        NumItemsT                num_items,
                                        int                      begin_bit = 0,
                                        int  end_bit = sizeof(KeyT) * 8,
                                        bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, begin_bit, end_bit, m_stream, debug_synchronous));
    }
};
}  // namespace muda
