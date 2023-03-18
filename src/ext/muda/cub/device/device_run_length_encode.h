#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_run_length_encode.cuh>
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_run_length_encode.html
class DeviceRunLengthEncode : public CubWrapper<DeviceRunLengthEncode>
{
  public:
    DeviceRunLengthEncode(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    template <typename InputIteratorT, typename UniqueOutputIteratorT, typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    DeviceRunLengthEncode& Encode(device_buffer<std::byte>& external_buffer,
                                  InputIteratorT            d_in,
                                  UniqueOutputIteratorT     d_unique_out,
                                  LengthsOutputIteratorT    d_counts_out,
                                  NumRunsOutputIteratorT    d_num_runs_out,
                                  int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRunLengthEncode::Encode(
            d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, m_stream, false));
    }

    template <typename InputIteratorT, typename OffsetsOutputIteratorT, typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    DeviceRunLengthEncode& NonTrivialRuns(device_buffer<std::byte>& external_buffer,
                                          InputIteratorT         d_in,
                                          OffsetsOutputIteratorT d_offsets_out,
                                          LengthsOutputIteratorT d_lengths_out,
                                          NumRunsOutputIteratorT d_num_runs_out,
                                          int                    num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRunLengthEncode::NonTrivialRuns(
            d_temp_storage, temp_storage_bytes, d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items, m_stream, false));
    }
};
}  // namespace muda
