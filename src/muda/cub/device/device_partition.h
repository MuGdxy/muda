#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_partition.cuh>
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_partition.html
class DevicePartition : public CubWrapper<DevicePartition>
{
  public:
    DevicePartition(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    template <typename InputIteratorT, typename FlagIterator, typename OutputIteratorT, typename NumSelectedIteratorT>
    DevicePartition& Flagged(DeviceBuffer<std::byte>& external_buffer,
                             InputIteratorT            d_in,
                             FlagIterator              d_flags,
                             OutputIteratorT           d_out,
                             NumSelectedIteratorT      d_num_selected_out,
                             int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DevicePartition::Flagged(
            d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, this->stream(), false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT, typename SelectOp>
    DevicePartition& If(DeviceBuffer<std::byte>& external_buffer,
                        InputIteratorT            d_in,
                        OutputIteratorT           d_out,
                        NumSelectedIteratorT      d_num_selected_out,
                        int                       num_items,
                        SelectOp                  select_op)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DevicePartition::If(
            d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, this->stream(), false));
    }

    template <typename InputIteratorT, typename FirstOutputIteratorT, typename SecondOutputIteratorT, typename UnselectedOutputIteratorT, typename NumSelectedIteratorT, typename SelectFirstPartOp, typename SelectSecondPartOp>
    DevicePartition& If(DeviceBuffer<std::byte>& external_buffer,
                        InputIteratorT            d_in,
                        FirstOutputIteratorT      d_first_part_out,
                        SecondOutputIteratorT     d_second_part_out,
                        UnselectedOutputIteratorT d_unselected_out,
                        NumSelectedIteratorT      d_num_selected_out,
                        int                       num_items,
                        SelectFirstPartOp         select_first_part_op,
                        SelectSecondPartOp        select_second_part_op)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DevicePartition::If(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_in,
                                                       d_first_part_out,
                                                       d_second_part_out,
                                                       d_unselected_out,
                                                       d_num_selected_out,
                                                       num_items,
                                                       select_first_part_op,
                                                       select_second_part_op,
                                                       this->stream(),
                                                       false));
    }
};
}  // namespace muda
