#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_merge_sort.cuh>
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_merge_sort.html
class DeviceMergeSort : public CubWrapper<DeviceMergeSort>
{
  public:
    DeviceMergeSort(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    template <typename KeyIteratorT, typename ValueIteratorT, typename OffsetT, typename CompareOpT>
    DeviceMergeSort& SortPairs(DeviceVector<std::byte>& external_buffer,
                               KeyIteratorT              d_keys,
                               ValueIteratorT            d_items,
                               OffsetT                   num_items,
                               CompareOpT                compare_op)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceMergeSort::SortPairs(
            d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, compare_op, this->stream(), false));
    }

    template <typename KeyInputIteratorT, typename ValueInputIteratorT, typename KeyIteratorT, typename ValueIteratorT, typename OffsetT, typename CompareOpT>
    DeviceMergeSort& SortPairsCopy(DeviceVector<std::byte>& external_buffer,
                                   KeyInputIteratorT         d_input_keys,
                                   ValueInputIteratorT       d_input_items,
                                   KeyIteratorT              d_output_keys,
                                   ValueIteratorT            d_output_items,
                                   OffsetT                   num_items,
                                   CompareOpT                compare_op)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceMergeSort::SortPairsCopy(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_input_keys,
                                                                  d_input_items,
                                                                  d_output_keys,
                                                                  d_output_items,
                                                                  num_items,
                                                                  compare_op,
                                                                  this->stream(),
                                                                  false));
    }

    template <typename KeyIteratorT, typename OffsetT, typename CompareOpT>
    DeviceMergeSort& SortKeys(DeviceVector<std::byte>& external_buffer,
                              KeyIteratorT              d_keys,
                              OffsetT                   num_items,
                              CompareOpT                compare_op)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceMergeSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, compare_op, this->stream(), false));
    }

    template <typename KeyInputIteratorT, typename KeyIteratorT, typename OffsetT, typename CompareOpT>
    DeviceMergeSort& SortKeysCopy(DeviceVector<std::byte>& external_buffer,
                                  KeyInputIteratorT         d_input_keys,
                                  KeyIteratorT              d_output_keys,
                                  OffsetT                   num_items,
                                  CompareOpT                compare_op)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceMergeSort::SortKeysCopy(
            d_temp_storage, temp_storage_bytes, d_input_keys, d_output_keys, num_items, compare_op, this->stream(), false));
    }

    template <typename KeyIteratorT, typename ValueIteratorT, typename OffsetT, typename CompareOpT>
    DeviceMergeSort& StableSortPairs(DeviceVector<std::byte>& external_buffer,
                                     KeyIteratorT              d_keys,
                                     ValueIteratorT            d_items,
                                     OffsetT                   num_items,
                                     CompareOpT                compare_op)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceMergeSort::StableSortPairs(
            d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, compare_op, this->stream(), false));
    }

    template <typename KeyIteratorT, typename OffsetT, typename CompareOpT>
    DeviceMergeSort& StableSortKeys(DeviceVector<std::byte>& external_buffer,
                                    KeyIteratorT              d_keys,
                                    OffsetT                   num_items,
                                    CompareOpT                compare_op)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceMergeSort::StableSortKeys(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, compare_op, this->stream(), false));
    }
};
}  // namespace muda
