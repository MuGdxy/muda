#pragma once
#include <cub/device/device_scan.cuh>
#ifdef __INTELLISENSE__
#include "device_scan.h"
#endif

template <typename T>
inline muda::DeviceScan& muda::DeviceScan::ExclusiveSum(
    device_buffer<std::byte>& external_buffer, T* d_out, T* d_in, int num_items)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes, d_in, d_out, num_items, stream_, false);
    prepareBuffer(external_buffer, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(
        external_buffer.data(), temp_storage_bytes, d_in, d_out, num_items, stream_, false);
    return *this;
}

template <typename T>
inline muda::DeviceScan& muda::DeviceScan::InclusiveSum(
    device_buffer<std::byte>& external_buffer, T* d_out, T* d_in, int num_items)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, d_in, d_out, num_items, stream_, false);
    prepareBuffer(external_buffer, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(
        external_buffer.data(), temp_storage_bytes, d_in, d_out, num_items, stream_, false);
    return *this;
}