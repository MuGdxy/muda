#pragma once
#include <cub/device/device_radix_sort.cuh>
#ifdef __INTELLISENSE__
#include "device_radix_sort.h"
#endif

template <typename KeyT, typename ValueT>
inline muda::DeviceRadixSort& muda::DeviceRadixSort::SortPairs(device_buffer<std::byte>& external_buffer,
                                                               KeyT* d_keys_out,
                                                               ValueT* d_values_out,
                                                               KeyT* d_keys_in,
                                                               ValueT* d_values_in,
                                                               int num_items,
                                                               int begin_bit,
                                                               int end_bit)
{
    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    temp_storage_bytes,
                                    d_keys_in,
                                    d_keys_out,
                                    d_values_in,
                                    d_values_out,
                                    num_items,
                                    begin_bit,
                                    end_bit,
                                    m_stream,
                                    false);
    // Allocate temporary storage
    prepareBuffer(external_buffer, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(external_buffer.data(),
                                    temp_storage_bytes,
                                    d_keys_in,
                                    d_keys_out,
                                    d_values_in,
                                    d_values_out,
                                    num_items,
                                    begin_bit,
                                    end_bit,
                                    m_stream,
                                    false);
    return *this;
}

template <typename KeyT>
inline muda::DeviceRadixSort& muda::DeviceRadixSort::SortKeys(device_buffer<std::byte>& external_buffer,
                                                              KeyT* d_keys_out,
                                                              KeyT* d_keys_in,
                                                              int   num_items,
                                                              int   begin_bit,
                                                              int   end_bit)
{
    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, m_stream, false);
    // Allocate temporary storage
    prepareBuffer(external_buffer, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(external_buffer.data(),
                                   temp_storage_bytes,
                                   d_keys_in,
                                   d_keys_out,
                                   num_items,
                                   begin_bit,
                                   end_bit,
                                   m_stream,
                                   false);
    return *this;
}
