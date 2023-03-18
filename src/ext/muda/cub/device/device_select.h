#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_select.cuh>
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_select.html
class DeviceSelect : public CubWrapper<DeviceSelect>
{
  public:
    DeviceSelect(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    template <typename InputIteratorT, typename FlagIterator, typename OutputIteratorT, typename NumSelectedIteratorT>
    DeviceSelect& Flagged(device_buffer<std::byte>& external_buffer,
                          InputIteratorT            d_in,
                          FlagIterator              d_flags,
                          OutputIteratorT           d_out,
                          NumSelectedIteratorT      d_num_selected_out,
                          int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSelect::Flagged(
            d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, m_stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT, typename SelectOp>
    DeviceSelect& If(device_buffer<std::byte>& external_buffer,
                     InputIteratorT            d_in,
                     OutputIteratorT           d_out,
                     NumSelectedIteratorT      d_num_selected_out,
                     int                       num_items,
                     SelectOp                  select_op)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSelect::If(
            d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, m_stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT>
    DeviceSelect& Unique(device_buffer<std::byte>& external_buffer,
                         InputIteratorT            d_in,
                         OutputIteratorT           d_out,
                         NumSelectedIteratorT      d_num_selected_out,
                         int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSelect::Unique(
            d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, m_stream, false));
    }
#if 0
    template <typename KeyInputIteratorT, typename ValueInputIteratorT, typename KeyOutputIteratorT, typename ValueOutputIteratorT, typename NumSelectedIteratorT>
    DeviceSelect& UniqueByKey(device_buffer<std::byte>& external_buffer,
                              KeyInputIteratorT         d_keys_in,
                              ValueInputIteratorT       d_values_in,
                              KeyOutputIteratorT        d_keys_out,
                              ValueOutputIteratorT      d_values_out,
                              NumSelectedIteratorT      d_num_selected_out,
                              int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSelect::UniqueByKey(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_keys_in,
                                                             d_values_in,
                                                             d_keys_out,
                                                             d_values_out,
                                                             d_num_selected_out,
                                                             num_items,
                                                             m_stream,
                                                             false));
    }
#endif
};
}  // namespace muda
