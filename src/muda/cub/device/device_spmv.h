#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_spmv.cuh>
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_spmv.html
class DeviceSpmv : public CubWrapper<DeviceSpmv>
{
  public:
    DeviceSpmv(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    // DeviceVector:

    template <typename ValueT>
    DeviceSpmv& CsrMV(DeviceVector<std::byte>& external_buffer,
                      const ValueT*            d_values,
                      const int*               d_row_offsets,
                      const int*               d_column_indices,
                      const ValueT*            d_vector_x,
                      ValueT*                  d_vector_y,
                      int                      num_rows,
                      int                      num_cols,
                      int                      num_nonzeros)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSpmv::CsrMV(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_values,
                                                     d_row_offsets,
                                                     d_column_indices,
                                                     d_vector_x,
                                                     d_vector_y,
                                                     num_rows,
                                                     num_cols,
                                                     num_nonzeros,
                                                     this->stream(),
                                                     false));
    }

    // DeviceBuffer:

    template <typename ValueT>
    DeviceSpmv& CsrMV(DeviceBuffer<std::byte>& external_buffer,
                      const ValueT*            d_values,
                      const int*               d_row_offsets,
                      const int*               d_column_indices,
                      const ValueT*            d_vector_x,
                      ValueT*                  d_vector_y,
                      int                      num_rows,
                      int                      num_cols,
                      int                      num_nonzeros)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSpmv::CsrMV(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_values,
                                                     d_row_offsets,
                                                     d_column_indices,
                                                     d_vector_x,
                                                     d_vector_y,
                                                     num_rows,
                                                     num_cols,
                                                     num_nonzeros,
                                                     this->stream(),
                                                     false));
    }

    // Origin:

    template <typename ValueT>
    DeviceSpmv& CsrMV(void*         d_temp_storage,
                      size_t&       temp_storage_bytes,
                      const ValueT* d_values,
                      const int*    d_row_offsets,
                      const int*    d_column_indices,
                      const ValueT* d_vector_x,
                      ValueT*       d_vector_y,
                      int           num_rows,
                      int           num_cols,
                      int           num_nonzeros)
    {
        checkCudaErrors(cub::DeviceSpmv::CsrMV(d_temp_storage,
                                               temp_storage_bytes,
                                               d_values,
                                               d_row_offsets,
                                               d_column_indices,
                                               d_vector_x,
                                               d_vector_y,
                                               num_rows,
                                               num_cols,
                                               num_nonzeros,
                                               this->stream(),
                                               false));
    }
};
}  // namespace muda
