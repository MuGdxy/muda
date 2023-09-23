#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_scan.cuh>
#else
namespace cub
{
class Equality
{
	//dummy class just for Intellisense
};
}
#endif


namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_scan.html
class DeviceScan : public CubWrapper<DeviceScan>
{
  public:
    DeviceScan(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceScan& ExclusiveSum(DeviceBuffer<std::byte>& external_buffer,
                             InputIteratorT            d_in,
                             OutputIteratorT           d_out,
                             int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT>
    DeviceScan& ExclusiveScan(DeviceBuffer<std::byte>& external_buffer,
                              InputIteratorT            d_in,
                              OutputIteratorT           d_out,
                              ScanOpT                   scan_op,
                              InitValueT                init_value,
                              int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceScan::ExclusiveScan(
            d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, init_value, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceScan& InclusiveSum(DeviceBuffer<std::byte>& external_buffer,
                             InputIteratorT            d_in,
                             OutputIteratorT           d_out,
                             int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT>
    DeviceScan& InclusiveScan(DeviceBuffer<std::byte>& external_buffer,
                              InputIteratorT            d_in,
                              OutputIteratorT           d_out,
                              ScanOpT                   scan_op,
                              int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceScan::InclusiveScan(
            d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, num_items, this->stream(), false));
    }

    template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename EqualityOpT = cub::Equality>
    DeviceScan& ExclusiveSumByKey(DeviceBuffer<std::byte>& external_buffer,
                                  KeysInputIteratorT        d_keys_in,
                                  ValuesInputIteratorT      d_values_in,
                                  ValuesOutputIteratorT     d_values_out,
                                  int                       num_items,
                                  EqualityOpT equality_op = EqualityOpT())
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceScan::ExclusiveSumByKey(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, num_items, equality_op, this->stream(), false));
    }

    template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename ScanOpT, typename InitValueT, typename EqualityOpT = cub::Equality>
    DeviceScan& ExclusiveScanByKey(DeviceBuffer<std::byte>& external_buffer,
                                   KeysInputIteratorT        d_keys_in,
                                   ValuesInputIteratorT      d_values_in,
                                   ValuesOutputIteratorT     d_values_out,
                                   ScanOpT                   scan_op,
                                   InitValueT                init_value,
                                   int                       num_items,
                                   EqualityOpT equality_op = EqualityOpT())
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceScan::ExclusiveScanByKey(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_keys_in,
                                                                  d_values_in,
                                                                  d_values_out,
                                                                  scan_op,
                                                                  init_value,
                                                                  num_items,
                                                                  equality_op,
                                                                  this->stream(),
                                                                  false));
    }

    template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename EqualityOpT = cub::Equality>
    DeviceScan& InclusiveSumByKey(DeviceBuffer<std::byte>& external_buffer,
                                  KeysInputIteratorT        d_keys_in,
                                  ValuesInputIteratorT      d_values_in,
                                  ValuesOutputIteratorT     d_values_out,
                                  int                       num_items,
                                  EqualityOpT equality_op = EqualityOpT())
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceScan::InclusiveSumByKey(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, num_items, equality_op, this->stream(), false));
    }

    template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename ScanOpT, typename EqualityOpT = cub::Equality>
    DeviceScan& InclusiveScanByKey(DeviceBuffer<std::byte>& external_buffer,
                                   KeysInputIteratorT        d_keys_in,
                                   ValuesInputIteratorT      d_values_in,
                                   ValuesOutputIteratorT     d_values_out,
                                   ScanOpT                   scan_op,
                                   int                       num_items,
                                   EqualityOpT equality_op = EqualityOpT())
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceScan::InclusiveScanByKey(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_keys_in,
                                                                  d_values_in,
                                                                  d_values_out,
                                                                  scan_op,
                                                                  num_items,
                                                                  equality_op,
                                                                  this->stream(),
                                                                  false));
    }
};
}  // namespace muda
