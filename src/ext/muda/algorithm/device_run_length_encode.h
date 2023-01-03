#pragma once
#include "base.h"

namespace muda
{
class DeviceRunLengthEncode : public AlgBase<DeviceRunLengthEncode>
{
  public:
    DeviceRunLengthEncode(cudaStream_t stream = nullptr)
        : AlgBase(stream)
    {
    }

    template <typename T>
    DeviceRunLengthEncode& Encode(device_buffer<std::byte>& external_buffer,
                                  T*                        d_unique_out,
                                  int*                      d_counts_out,
                                  int*                      d_num_runs_out,
                                  T*                        d_in,
                                  int                       num_items);

    template <typename T>
    DeviceRunLengthEncode& Encode(
        T* d_unique_out, int* d_counts_out, int* d_num_runs_out, T* d_in, int num_items)
    {
        device_buffer<std::byte> external_buffer;
        Encode(external_buffer, d_unique_out, d_counts_out, d_num_runs_out, d_in, num_items);
    }
};
}  // namespace muda

#ifndef __INTELLISENSE__
#include "device_run_length_encode.inl"
#endif
