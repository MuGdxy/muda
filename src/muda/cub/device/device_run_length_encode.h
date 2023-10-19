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

    // DeviceVector:

    template <typename InputIteratorT, typename UniqueOutputIteratorT, typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    DeviceRunLengthEncode& Encode(DeviceVector<std::byte>& external_buffer,
                                  InputIteratorT           d_in,
                                  UniqueOutputIteratorT    d_unique_out,
                                  LengthsOutputIteratorT   d_counts_out,
                                  NumRunsOutputIteratorT   d_num_runs_out,
                                  int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_in,
                                                                 d_unique_out,
                                                                 d_counts_out,
                                                                 d_num_runs_out,
                                                                 num_items,
                                                                 this->stream(),
                                                                 false));
    }

    template <typename InputIteratorT, typename OffsetsOutputIteratorT, typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    DeviceRunLengthEncode& NonTrivialRuns(DeviceVector<std::byte>& external_buffer,
                                          InputIteratorT         d_in,
                                          OffsetsOutputIteratorT d_offsets_out,
                                          LengthsOutputIteratorT d_lengths_out,
                                          NumRunsOutputIteratorT d_num_runs_out,
                                          int                    num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(
            cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_in,
                                                       d_offsets_out,
                                                       d_lengths_out,
                                                       d_num_runs_out,
                                                       num_items,
                                                       this->stream(),
                                                       false));
    }

    // DeviceBuffer:

    template <typename InputIteratorT, typename UniqueOutputIteratorT, typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    DeviceRunLengthEncode& Encode(DeviceBuffer<std::byte>& external_buffer,
                                  InputIteratorT           d_in,
                                  UniqueOutputIteratorT    d_unique_out,
                                  LengthsOutputIteratorT   d_counts_out,
                                  NumRunsOutputIteratorT   d_num_runs_out,
                                  int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_in,
                                                                 d_unique_out,
                                                                 d_counts_out,
                                                                 d_num_runs_out,
                                                                 num_items,
                                                                 this->stream(),
                                                                 false));
    }

    template <typename InputIteratorT, typename OffsetsOutputIteratorT, typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    DeviceRunLengthEncode& NonTrivialRuns(DeviceBuffer<std::byte>& external_buffer,
                                          InputIteratorT         d_in,
                                          OffsetsOutputIteratorT d_offsets_out,
                                          LengthsOutputIteratorT d_lengths_out,
                                          NumRunsOutputIteratorT d_num_runs_out,
                                          int                    num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(
            cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_in,
                                                       d_offsets_out,
                                                       d_lengths_out,
                                                       d_num_runs_out,
                                                       num_items,
                                                       this->stream(),
                                                       false));
    }

    // Origin:

    template <typename InputIteratorT, typename UniqueOutputIteratorT, typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    DeviceRunLengthEncode& Encode(void*                  d_temp_storage,
                                  size_t&                temp_storage_bytes,
                                  InputIteratorT         d_in,
                                  UniqueOutputIteratorT  d_unique_out,
                                  LengthsOutputIteratorT d_counts_out,
                                  NumRunsOutputIteratorT d_num_runs_out,
                                  int                    num_items)
    {
        checkCudaErrors(cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_in,
                                                                 d_unique_out,
                                                                 d_counts_out,
                                                                 d_num_runs_out,
                                                                 num_items,
                                                                 this->stream(),
                                                                 false));
    }

    template <typename InputIteratorT, typename OffsetsOutputIteratorT, typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    DeviceRunLengthEncode& NonTrivialRuns(void*          d_temp_storage,
                                          size_t&        temp_storage_bytes,
                                          InputIteratorT d_in,
                                          OffsetsOutputIteratorT d_offsets_out,
                                          LengthsOutputIteratorT d_lengths_out,
                                          NumRunsOutputIteratorT d_num_runs_out,
                                          int                    num_items)
    {
        checkCudaErrors(
            cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_in,
                                                       d_offsets_out,
                                                       d_lengths_out,
                                                       d_num_runs_out,
                                                       num_items,
                                                       this->stream(),
                                                       false));
    }
};
}  // namespace muda
