#pragma once
#include <cub/device/device_reduce.cuh>
#ifdef __INTELLISENSE__
#include "device_reduce.h"
#endif

template <typename T, typename Compare>
muda::DeviceReduce& muda::DeviceReduce::Reduce(device_buffer<std::byte>& external_buffer,
                                               T*      d_out,
                                               T*      d_in,
                                               int     num_items,
                                               Compare cmp,
                                               T       init)
{
    void*  d_temp_storage     = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(
        nullptr, temp_storage_bytes, d_in, d_out, num_items, cmp, init, m_stream, false);
    prepareBuffer(external_buffer, temp_storage_bytes);
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, cmp, init, m_stream, false);
    return *this;
}

template <typename T>
inline muda::DeviceReduce& muda::DeviceReduce::Max(
    device_buffer<std::byte>& external_buffer, T* d_out, T* d_in, int num_items)
{
    size_t    temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, temp_storage_bytes, d_in, d_out, num_items);
    // Allocate temporary storage
    prepareBuffer(external_buffer, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Max(external_buffer.data(), temp_storage_bytes, d_in, d_out, num_items, m_stream, false);
    return *this;
}
