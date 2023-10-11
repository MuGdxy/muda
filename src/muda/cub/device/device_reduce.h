#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_reduce.cuh>
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html
class DeviceReduce : public CubWrapper<DeviceReduce>
{
  public:
    DeviceReduce(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }
    // for DeviceVector:
    template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT, typename T>
    DeviceReduce& Reduce(DeviceVector<std::byte>& external_buffer,
                         InputIteratorT           d_in,
                         OutputIteratorT          d_out,
                         int                      num_items,
                         ReductionOpT             reduction_op,
                         T                        init)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::Reduce(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, init, this->stream(), false));
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& Sum(DeviceVector<std::byte>& external_buffer,
                      InputIteratorT           d_in,
                      OutputIteratorT          d_out,
                      int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& Min(DeviceVector<std::byte>& external_buffer,
                      InputIteratorT           d_in,
                      OutputIteratorT          d_out,
                      int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::Min(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& ArgMin(DeviceVector<std::byte>& external_buffer,
                         InputIteratorT           d_in,
                         OutputIteratorT          d_out,
                         int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::ArgMin(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& Max(DeviceVector<std::byte>& external_buffer,
                      InputIteratorT           d_in,
                      OutputIteratorT          d_out,
                      int                      num_items)
    {

        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::Max(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& ArgMax(DeviceVector<std::byte>& external_buffer,
                         InputIteratorT           d_in,
                         OutputIteratorT          d_out,
                         int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::ArgMax(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }

    template <typename KeysInputIteratorT, typename UniqueOutputIteratorT, typename ValuesInputIteratorT, typename AggregatesOutputIteratorT, typename NumRunsOutputIteratorT, typename ReductionOpT>
    DeviceReduce& ReduceByKey(DeviceVector<std::byte>&  external_buffer,
                              KeysInputIteratorT        d_keys_in,
                              UniqueOutputIteratorT     d_unique_out,
                              ValuesInputIteratorT      d_values_in,
                              AggregatesOutputIteratorT d_aggregates_out,
                              NumRunsOutputIteratorT    d_num_runs_out,
                              ReductionOpT              reduction_op,
                              int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::ReduceByKey(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_keys_in,
                                                             d_unique_out,
                                                             d_values_in,
                                                             d_aggregates_out,
                                                             d_num_runs_out,
                                                             reduction_op,
                                                             num_items));
    }

    // for DeviceBuffer
    template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT, typename T>
    DeviceReduce& Reduce(DeviceBuffer<std::byte>& external_buffer,
                         InputIteratorT           d_in,
                         OutputIteratorT          d_out,
                         int                      num_items,
                         ReductionOpT             reduction_op,
                         T                        init)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::Reduce(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, init, this->stream(), false));
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& Sum(DeviceBuffer<std::byte>& external_buffer,
                      InputIteratorT           d_in,
                      OutputIteratorT          d_out,
                      int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& Min(DeviceBuffer<std::byte>& external_buffer,
                      InputIteratorT           d_in,
                      OutputIteratorT          d_out,
                      int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::Min(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& ArgMin(DeviceBuffer<std::byte>& external_buffer,
                         InputIteratorT           d_in,
                         OutputIteratorT          d_out,
                         int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::ArgMin(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& Max(DeviceBuffer<std::byte>& external_buffer,
                      InputIteratorT           d_in,
                      OutputIteratorT          d_out,
                      int                      num_items)
    {

        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::Max(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceReduce& ArgMax(DeviceBuffer<std::byte>& external_buffer,
                         InputIteratorT           d_in,
                         OutputIteratorT          d_out,
                         int                      num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::ArgMax(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, this->stream(), false));
    }

    template <typename KeysInputIteratorT, typename UniqueOutputIteratorT, typename ValuesInputIteratorT, typename AggregatesOutputIteratorT, typename NumRunsOutputIteratorT, typename ReductionOpT>
    DeviceReduce& ReduceByKey(DeviceBuffer<std::byte>&  external_buffer,
                              KeysInputIteratorT        d_keys_in,
                              UniqueOutputIteratorT     d_unique_out,
                              ValuesInputIteratorT      d_values_in,
                              AggregatesOutputIteratorT d_aggregates_out,
                              NumRunsOutputIteratorT    d_num_runs_out,
                              ReductionOpT              reduction_op,
                              int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceReduce::ReduceByKey(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_keys_in,
                                                             d_unique_out,
                                                             d_values_in,
                                                             d_aggregates_out,
                                                             d_num_runs_out,
                                                             reduction_op,
                                                             num_items));
    }
};
}  // namespace muda