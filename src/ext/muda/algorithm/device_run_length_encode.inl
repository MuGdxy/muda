#pragma once
#include <cub/device/device_run_length_encode.cuh>
#ifdef __INTELLISENSE__
#include "device_run_length_encode.h"
#endif


namespace muda
{
template <typename T>
DeviceRunLengthEncode& DeviceRunLengthEncode::Encode(device_buffer<std::byte>& external_buffer,
                                                     T*   d_unique_out,
                                                     int* d_counts_out,
                                                     int* d_num_runs_out,
                                                     T*   d_in,
                                                     int  num_items)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(external_buffer.data(),
                                       temp_storage_bytes,
                                       d_in,
                                       d_unique_out,
                                       d_counts_out,
                                       d_num_runs_out,
                                       num_items,
                                       m_stream,
                                       false);
    prepareBuffer(external_buffer, temp_storage_bytes);
    cub::DeviceRunLengthEncode::Encode(external_buffer.data(),
                                       temp_storage_bytes,
                                       d_in,
                                       d_unique_out,
                                       d_counts_out,
                                       d_num_runs_out,
                                       num_items,
                                       m_stream,
                                       false);
    return *this;
}

}  // namespace muda
