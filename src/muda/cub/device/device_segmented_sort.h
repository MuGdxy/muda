#pragma once
#include <muda/cub/device/cub_wrapper.h>
#include "details/cub_wrapper_macro_def.inl"
#ifndef __INTELLISENSE__
#include <cub/device/device_segmented_sort.cuh>
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_segmented_reduce.html
class DeviceSegmentedSort : public CubWrapper<DeviceSegmentedSort>
{
  public:
    DeviceSegmentedSort(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    // DeviceVector:

    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeys(DeviceVector<std::byte>& external_buffer,
                                  const KeyT*              d_keys_in,
                                  KeyT*                    d_keys_out,
                                  int                      num_items,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortKeys(d_temp_storage,
                                            temp_storage_bytes,
                                            d_keys_in,
                                            d_keys_out,
                                            num_items,
                                            num_segments,
                                            d_begin_offsets,
                                            d_end_offsets,
                                            _stream,
                                            false));
    }

    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeysDescending(DeviceVector<std::byte>& external_buffer,
                                            const KeyT* d_keys_in,
                                            KeyT*       d_keys_out,
                                            int         num_items,
                                            int         num_segments,
                                            BeginOffsetIteratorT d_begin_offsets,
                                            EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortKeysDescending(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_keys_in,
                                                      d_keys_out,
                                                      num_items,
                                                      num_segments,
                                                      d_begin_offsets,
                                                      d_end_offsets,
                                                      _stream,
                                                      false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeys(DeviceVector<std::byte>& external_buffer,
                                  cub::DoubleBuffer<KeyT>& d_keys,
                                  int                      num_items,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortKeys(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeysDescending(DeviceVector<std::byte>& external_buffer,
                                            cub::DoubleBuffer<KeyT>& d_keys,
                                            int                      num_items,
                                            int num_segments,
                                            BeginOffsetIteratorT d_begin_offsets,
                                            EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortKeysDescending(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeys(DeviceVector<std::byte>& external_buffer,
                                        const KeyT*          d_keys_in,
                                        KeyT*                d_keys_out,
                                        int                  num_items,
                                        int                  num_segments,
                                        BeginOffsetIteratorT d_begin_offsets,
                                        EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortKeys(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_keys_in,
                                                  d_keys_out,
                                                  num_items,
                                                  num_segments,
                                                  d_begin_offsets,
                                                  d_end_offsets,
                                                  _stream,
                                                  false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeysDescending(DeviceVector<std::byte>& external_buffer,
                                                  const KeyT* d_keys_in,
                                                  KeyT*       d_keys_out,
                                                  int         num_items,
                                                  int         num_segments,
                                                  BeginOffsetIteratorT d_begin_offsets,
                                                  EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortKeysDescending(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_keys_in,
                                                            d_keys_out,
                                                            num_items,
                                                            num_segments,
                                                            d_begin_offsets,
                                                            d_end_offsets,
                                                            _stream,
                                                            false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeys(DeviceVector<std::byte>& external_buffer,
                                        cub::DoubleBuffer<KeyT>& d_keys,
                                        int                      num_items,
                                        int                      num_segments,
                                        BeginOffsetIteratorT d_begin_offsets,
                                        EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortKeys(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeysDescending(DeviceVector<std::byte>& external_buffer,
                                                  cub::DoubleBuffer<KeyT>& d_keys,
                                                  int num_items,
                                                  int num_segments,
                                                  BeginOffsetIteratorT d_begin_offsets,
                                                  EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortKeysDescending(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairs(DeviceVector<std::byte>& external_buffer,
                                   const KeyT*              d_keys_in,
                                   KeyT*                    d_keys_out,
                                   const ValueT*            d_values_in,
                                   ValueT*                  d_values_out,
                                   int                      num_items,
                                   int                      num_segments,
                                   BeginOffsetIteratorT     d_begin_offsets,
                                   EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortPairs(d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_in,
                                             d_keys_out,
                                             d_values_in,
                                             d_values_out,
                                             num_items,
                                             num_segments,
                                             d_begin_offsets,
                                             d_end_offsets,
                                             _stream,
                                             false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairsDescending(DeviceVector<std::byte>& external_buffer,
                                             const KeyT*   d_keys_in,
                                             KeyT*         d_keys_out,
                                             const ValueT* d_values_in,
                                             ValueT*       d_values_out,
                                             int           num_items,
                                             int           num_segments,
                                             BeginOffsetIteratorT d_begin_offsets,
                                             EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortPairsDescending(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys_in,
                                                       d_keys_out,
                                                       d_values_in,
                                                       d_values_out,
                                                       num_items,
                                                       num_segments,
                                                       d_begin_offsets,
                                                       d_end_offsets,
                                                       _stream,
                                                       false));
    }

    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairs(DeviceVector<std::byte>&   external_buffer,
                                   cub::DoubleBuffer<KeyT>&   d_keys,
                                   cub::DoubleBuffer<ValueT>& d_values,
                                   int                        num_items,
                                   int                        num_segments,
                                   BeginOffsetIteratorT       d_begin_offsets,
                                   EndOffsetIteratorT         d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortPairs(d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys,
                                             d_values,
                                             num_items,
                                             num_segments,
                                             d_begin_offsets,
                                             d_end_offsets,
                                             _stream,
                                             false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairsDescending(DeviceVector<std::byte>& external_buffer,
                                             cub::DoubleBuffer<KeyT>& d_keys,
                                             cub::DoubleBuffer<ValueT>& d_values,
                                             int num_items,
                                             int num_segments,
                                             BeginOffsetIteratorT d_begin_offsets,
                                             EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortPairsDescending(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys,
                                                       d_values,
                                                       num_items,
                                                       num_segments,
                                                       d_begin_offsets,
                                                       d_end_offsets,
                                                       _stream,
                                                       false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairs(DeviceVector<std::byte>& external_buffer,
                                         const KeyT*          d_keys_in,
                                         KeyT*                d_keys_out,
                                         const ValueT*        d_values_in,
                                         ValueT*              d_values_out,
                                         int                  num_items,
                                         int                  num_segments,
                                         BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortPairs(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_keys_in,
                                                   d_keys_out,
                                                   d_values_in,
                                                   d_values_out,
                                                   num_items,
                                                   num_segments,
                                                   d_begin_offsets,
                                                   d_end_offsets,
                                                   _stream,
                                                   false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairsDescending(DeviceVector<std::byte>& external_buffer,
                                                   const KeyT*   d_keys_in,
                                                   KeyT*         d_keys_out,
                                                   const ValueT* d_values_in,
                                                   ValueT*       d_values_out,
                                                   int           num_items,
                                                   int           num_segments,
                                                   BeginOffsetIteratorT d_begin_offsets,
                                                   EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortPairsDescending(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_keys_in,
                                                             d_keys_out,
                                                             d_values_in,
                                                             d_values_out,
                                                             num_items,
                                                             num_segments,
                                                             d_begin_offsets,
                                                             d_end_offsets,
                                                             _stream,
                                                             false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairs(DeviceVector<std::byte>& external_buffer,
                                         cub::DoubleBuffer<KeyT>&   d_keys,
                                         cub::DoubleBuffer<ValueT>& d_values,
                                         int                        num_items,
                                         int                  num_segments,
                                         BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortPairs(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_keys,
                                                   d_values,
                                                   num_items,
                                                   num_segments,
                                                   d_begin_offsets,
                                                   d_end_offsets,
                                                   _stream,
                                                   false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairsDescending(DeviceVector<std::byte>& external_buffer,
                                                   cub::DoubleBuffer<KeyT>& d_keys,
                                                   cub::DoubleBuffer<ValueT>& d_values,
                                                   int num_items,
                                                   int num_segments,
                                                   BeginOffsetIteratorT d_begin_offsets,
                                                   EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortPairsDescending(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_keys,
                                                             d_values,
                                                             num_items,
                                                             num_segments,
                                                             d_begin_offsets,
                                                             d_end_offsets,
                                                             _stream,
                                                             false));
    }

    // DeviceBuffer:

    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeys(DeviceBuffer<std::byte>& external_buffer,
                                  const KeyT*              d_keys_in,
                                  KeyT*                    d_keys_out,
                                  int                      num_items,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortKeys(d_temp_storage,
                                            temp_storage_bytes,
                                            d_keys_in,
                                            d_keys_out,
                                            num_items,
                                            num_segments,
                                            d_begin_offsets,
                                            d_end_offsets,
                                            _stream,
                                            false));
    }

    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeysDescending(DeviceBuffer<std::byte>& external_buffer,
                                            const KeyT* d_keys_in,
                                            KeyT*       d_keys_out,
                                            int         num_items,
                                            int         num_segments,
                                            BeginOffsetIteratorT d_begin_offsets,
                                            EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortKeysDescending(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_keys_in,
                                                      d_keys_out,
                                                      num_items,
                                                      num_segments,
                                                      d_begin_offsets,
                                                      d_end_offsets,
                                                      _stream,
                                                      false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeys(DeviceBuffer<std::byte>& external_buffer,
                                  cub::DoubleBuffer<KeyT>& d_keys,
                                  int                      num_items,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortKeys(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeysDescending(DeviceBuffer<std::byte>& external_buffer,
                                            cub::DoubleBuffer<KeyT>& d_keys,
                                            int                      num_items,
                                            int num_segments,
                                            BeginOffsetIteratorT d_begin_offsets,
                                            EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortKeysDescending(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeys(DeviceBuffer<std::byte>& external_buffer,
                                        const KeyT*          d_keys_in,
                                        KeyT*                d_keys_out,
                                        int                  num_items,
                                        int                  num_segments,
                                        BeginOffsetIteratorT d_begin_offsets,
                                        EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortKeys(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_keys_in,
                                                  d_keys_out,
                                                  num_items,
                                                  num_segments,
                                                  d_begin_offsets,
                                                  d_end_offsets,
                                                  _stream,
                                                  false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeysDescending(DeviceBuffer<std::byte>& external_buffer,
                                                  const KeyT* d_keys_in,
                                                  KeyT*       d_keys_out,
                                                  int         num_items,
                                                  int         num_segments,
                                                  BeginOffsetIteratorT d_begin_offsets,
                                                  EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortKeysDescending(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_keys_in,
                                                            d_keys_out,
                                                            num_items,
                                                            num_segments,
                                                            d_begin_offsets,
                                                            d_end_offsets,
                                                            _stream,
                                                            false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeys(DeviceBuffer<std::byte>& external_buffer,
                                        cub::DoubleBuffer<KeyT>& d_keys,
                                        int                      num_items,
                                        int                      num_segments,
                                        BeginOffsetIteratorT d_begin_offsets,
                                        EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortKeys(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeysDescending(DeviceBuffer<std::byte>& external_buffer,
                                                  cub::DoubleBuffer<KeyT>& d_keys,
                                                  int num_items,
                                                  int num_segments,
                                                  BeginOffsetIteratorT d_begin_offsets,
                                                  EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortKeysDescending(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairs(DeviceBuffer<std::byte>& external_buffer,
                                   const KeyT*              d_keys_in,
                                   KeyT*                    d_keys_out,
                                   const ValueT*            d_values_in,
                                   ValueT*                  d_values_out,
                                   int                      num_items,
                                   int                      num_segments,
                                   BeginOffsetIteratorT     d_begin_offsets,
                                   EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortPairs(d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_in,
                                             d_keys_out,
                                             d_values_in,
                                             d_values_out,
                                             num_items,
                                             num_segments,
                                             d_begin_offsets,
                                             d_end_offsets,
                                             _stream,
                                             false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairsDescending(DeviceBuffer<std::byte>& external_buffer,
                                             const KeyT*   d_keys_in,
                                             KeyT*         d_keys_out,
                                             const ValueT* d_values_in,
                                             ValueT*       d_values_out,
                                             int           num_items,
                                             int           num_segments,
                                             BeginOffsetIteratorT d_begin_offsets,
                                             EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortPairsDescending(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys_in,
                                                       d_keys_out,
                                                       d_values_in,
                                                       d_values_out,
                                                       num_items,
                                                       num_segments,
                                                       d_begin_offsets,
                                                       d_end_offsets,
                                                       _stream,
                                                       false));
    }

    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairs(DeviceBuffer<std::byte>&   external_buffer,
                                   cub::DoubleBuffer<KeyT>&   d_keys,
                                   cub::DoubleBuffer<ValueT>& d_values,
                                   int                        num_items,
                                   int                        num_segments,
                                   BeginOffsetIteratorT       d_begin_offsets,
                                   EndOffsetIteratorT         d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortPairs(d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys,
                                             d_values,
                                             num_items,
                                             num_segments,
                                             d_begin_offsets,
                                             d_end_offsets,
                                             _stream,
                                             false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairsDescending(DeviceBuffer<std::byte>& external_buffer,
                                             cub::DoubleBuffer<KeyT>& d_keys,
                                             cub::DoubleBuffer<ValueT>& d_values,
                                             int num_items,
                                             int num_segments,
                                             BeginOffsetIteratorT d_begin_offsets,
                                             EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::SortPairsDescending(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys,
                                                       d_values,
                                                       num_items,
                                                       num_segments,
                                                       d_begin_offsets,
                                                       d_end_offsets,
                                                       _stream,
                                                       false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairs(DeviceBuffer<std::byte>& external_buffer,
                                         const KeyT*          d_keys_in,
                                         KeyT*                d_keys_out,
                                         const ValueT*        d_values_in,
                                         ValueT*              d_values_out,
                                         int                  num_items,
                                         int                  num_segments,
                                         BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortPairs(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_keys_in,
                                                   d_keys_out,
                                                   d_values_in,
                                                   d_values_out,
                                                   num_items,
                                                   num_segments,
                                                   d_begin_offsets,
                                                   d_end_offsets,
                                                   _stream,
                                                   false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairsDescending(DeviceBuffer<std::byte>& external_buffer,
                                                   const KeyT*   d_keys_in,
                                                   KeyT*         d_keys_out,
                                                   const ValueT* d_values_in,
                                                   ValueT*       d_values_out,
                                                   int           num_items,
                                                   int           num_segments,
                                                   BeginOffsetIteratorT d_begin_offsets,
                                                   EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortPairsDescending(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_keys_in,
                                                             d_keys_out,
                                                             d_values_in,
                                                             d_values_out,
                                                             num_items,
                                                             num_segments,
                                                             d_begin_offsets,
                                                             d_end_offsets,
                                                             _stream,
                                                             false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairs(DeviceBuffer<std::byte>& external_buffer,
                                         cub::DoubleBuffer<KeyT>&   d_keys,
                                         cub::DoubleBuffer<ValueT>& d_values,
                                         int                        num_items,
                                         int                  num_segments,
                                         BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortPairs(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_keys,
                                                   d_values,
                                                   num_items,
                                                   num_segments,
                                                   d_begin_offsets,
                                                   d_end_offsets,
                                                   _stream,
                                                   false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairsDescending(DeviceBuffer<std::byte>& external_buffer,
                                                   cub::DoubleBuffer<KeyT>& d_keys,
                                                   cub::DoubleBuffer<ValueT>& d_values,
                                                   int num_items,
                                                   int num_segments,
                                                   BeginOffsetIteratorT d_begin_offsets,
                                                   EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::StableSortPairsDescending(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_keys,
                                                             d_values,
                                                             num_items,
                                                             num_segments,
                                                             d_begin_offsets,
                                                             d_end_offsets,
                                                             _stream,
                                                             false));
    }

    // Origin:

    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeys(void*                d_temp_storage,
                                  size_t&              temp_storage_bytes,
                                  const KeyT*          d_keys_in,
                                  KeyT*                d_keys_out,
                                  int                  num_items,
                                  int                  num_segments,
                                  BeginOffsetIteratorT d_begin_offsets,
                                  EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::SortKeys(d_temp_storage,
                                                              temp_storage_bytes,
                                                              d_keys_in,
                                                              d_keys_out,
                                                              num_items,
                                                              num_segments,
                                                              d_begin_offsets,
                                                              d_end_offsets,
                                                              _stream,
                                                              false));
    }

    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeysDescending(void*       d_temp_storage,
                                            size_t&     temp_storage_bytes,
                                            const KeyT* d_keys_in,
                                            KeyT*       d_keys_out,
                                            int         num_items,
                                            int         num_segments,
                                            BeginOffsetIteratorT d_begin_offsets,
                                            EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(
            cub::SortKeysDescending(d_temp_storage,
                                    temp_storage_bytes,
                                    d_keys_in,
                                    d_keys_out,
                                    num_items,
                                    num_segments,
                                    d_begin_offsets,
                                    d_end_offsets,
                                    _stream,
                                    false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeys(void*                    d_temp_storage,
                                  size_t&                  temp_storage_bytes,
                                  cub::DoubleBuffer<KeyT>& d_keys,
                                  int                      num_items,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::SortKeys(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortKeysDescending(void*   d_temp_storage,
                                            size_t& temp_storage_bytes,
                                            cub::DoubleBuffer<KeyT>& d_keys,
                                            int                      num_items,
                                            int num_segments,
                                            BeginOffsetIteratorT d_begin_offsets,
                                            EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::SortKeysDescending(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeys(void*                d_temp_storage,
                                        size_t&              temp_storage_bytes,
                                        const KeyT*          d_keys_in,
                                        KeyT*                d_keys_out,
                                        int                  num_items,
                                        int                  num_segments,
                                        BeginOffsetIteratorT d_begin_offsets,
                                        EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::StableSortKeys(d_temp_storage,
                                                                    temp_storage_bytes,
                                                                    d_keys_in,
                                                                    d_keys_out,
                                                                    num_items,
                                                                    num_segments,
                                                                    d_begin_offsets,
                                                                    d_end_offsets,
                                                                    _stream,
                                                                    false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeysDescending(void*   d_temp_storage,
                                                  size_t& temp_storage_bytes,
                                                  const KeyT* d_keys_in,
                                                  KeyT*       d_keys_out,
                                                  int         num_items,
                                                  int         num_segments,
                                                  BeginOffsetIteratorT d_begin_offsets,
                                                  EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(
            cub::StableSortKeysDescending(d_temp_storage,
                                          temp_storage_bytes,
                                          d_keys_in,
                                          d_keys_out,
                                          num_items,
                                          num_segments,
                                          d_begin_offsets,
                                          d_end_offsets,
                                          _stream,
                                          false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeys(void*   d_temp_storage,
                                        size_t& temp_storage_bytes,
                                        cub::DoubleBuffer<KeyT>& d_keys,
                                        int                      num_items,
                                        int                      num_segments,
                                        BeginOffsetIteratorT d_begin_offsets,
                                        EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::StableSortKeys(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortKeysDescending(void*   d_temp_storage,
                                                  size_t& temp_storage_bytes,
                                                  cub::DoubleBuffer<KeyT>& d_keys,
                                                  int num_items,
                                                  int num_segments,
                                                  BeginOffsetIteratorT d_begin_offsets,
                                                  EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::StableSortKeysDescending(
            d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairs(void*                d_temp_storage,
                                   size_t&              temp_storage_bytes,
                                   const KeyT*          d_keys_in,
                                   KeyT*                d_keys_out,
                                   const ValueT*        d_values_in,
                                   ValueT*              d_values_out,
                                   int                  num_items,
                                   int                  num_segments,
                                   BeginOffsetIteratorT d_begin_offsets,
                                   EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::SortPairs(d_temp_storage,
                                                               temp_storage_bytes,
                                                               d_keys_in,
                                                               d_keys_out,
                                                               d_values_in,
                                                               d_values_out,
                                                               num_items,
                                                               num_segments,
                                                               d_begin_offsets,
                                                               d_end_offsets,
                                                               _stream,
                                                               false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairsDescending(void*         d_temp_storage,
                                             size_t&       temp_storage_bytes,
                                             const KeyT*   d_keys_in,
                                             KeyT*         d_keys_out,
                                             const ValueT* d_values_in,
                                             ValueT*       d_values_out,
                                             int           num_items,
                                             int           num_segments,
                                             BeginOffsetIteratorT d_begin_offsets,
                                             EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(
            cub::SortPairsDescending(d_temp_storage,
                                     temp_storage_bytes,
                                     d_keys_in,
                                     d_keys_out,
                                     d_values_in,
                                     d_values_out,
                                     num_items,
                                     num_segments,
                                     d_begin_offsets,
                                     d_end_offsets,
                                     _stream,
                                     false));
    }

    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairs(void*                    d_temp_storage,
                                   size_t&                  temp_storage_bytes,
                                   cub::DoubleBuffer<KeyT>& d_keys,
                                   cub::DoubleBuffer<ValueT>& d_values,
                                   int                        num_items,
                                   int                        num_segments,
                                   BeginOffsetIteratorT       d_begin_offsets,
                                   EndOffsetIteratorT         d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::SortPairs(d_temp_storage,
                                                               temp_storage_bytes,
                                                               d_keys,
                                                               d_values,
                                                               num_items,
                                                               num_segments,
                                                               d_begin_offsets,
                                                               d_end_offsets,
                                                               _stream,
                                                               false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& SortPairsDescending(void*   d_temp_storage,
                                             size_t& temp_storage_bytes,
                                             cub::DoubleBuffer<KeyT>& d_keys,
                                             cub::DoubleBuffer<ValueT>& d_values,
                                             int num_items,
                                             int num_segments,
                                             BeginOffsetIteratorT d_begin_offsets,
                                             EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairs(void*         d_temp_storage,
                                         size_t&       temp_storage_bytes,
                                         const KeyT*   d_keys_in,
                                         KeyT*         d_keys_out,
                                         const ValueT* d_values_in,
                                         ValueT*       d_values_out,
                                         int           num_items,
                                         int           num_segments,
                                         BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::StableSortPairs(d_temp_storage,
                                                                     temp_storage_bytes,
                                                                     d_keys_in,
                                                                     d_keys_out,
                                                                     d_values_in,
                                                                     d_values_out,
                                                                     num_items,
                                                                     num_segments,
                                                                     d_begin_offsets,
                                                                     d_end_offsets,
                                                                     _stream,
                                                                     false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairsDescending(void*   d_temp_storage,
                                                   size_t& temp_storage_bytes,
                                                   const KeyT*   d_keys_in,
                                                   KeyT*         d_keys_out,
                                                   const ValueT* d_values_in,
                                                   ValueT*       d_values_out,
                                                   int           num_items,
                                                   int           num_segments,
                                                   BeginOffsetIteratorT d_begin_offsets,
                                                   EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(
            cub::StableSortPairsDescending(d_temp_storage,
                                           temp_storage_bytes,
                                           d_keys_in,
                                           d_keys_out,
                                           d_values_in,
                                           d_values_out,
                                           num_items,
                                           num_segments,
                                           d_begin_offsets,
                                           d_end_offsets,
                                           _stream,
                                           false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairs(void*   d_temp_storage,
                                         size_t& temp_storage_bytes,
                                         cub::DoubleBuffer<KeyT>&   d_keys,
                                         cub::DoubleBuffer<ValueT>& d_values,
                                         int                        num_items,
                                         int                  num_segments,
                                         BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::StableSortPairs(d_temp_storage,
                                                                     temp_storage_bytes,
                                                                     d_keys,
                                                                     d_values,
                                                                     num_items,
                                                                     num_segments,
                                                                     d_begin_offsets,
                                                                     d_end_offsets,
                                                                     _stream,
                                                                     false));
    }


    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedSort& StableSortPairsDescending(void*   d_temp_storage,
                                                   size_t& temp_storage_bytes,
                                                   cub::DoubleBuffer<KeyT>& d_keys,
                                                   cub::DoubleBuffer<ValueT>& d_values,
                                                   int num_items,
                                                   int num_segments,
                                                   BeginOffsetIteratorT d_begin_offsets,
                                                   EndOffsetIteratorT d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::StableSortPairsDescending(
            d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }
};
}  // namespace muda

#include "details/cub_wrapper_macro_undef.inl"