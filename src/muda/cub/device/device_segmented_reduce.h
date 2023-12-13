#pragma once
#include <muda/cub/device/cub_wrapper.h>
#include "details/cub_wrapper_macro_def.inl"
#ifndef __INTELLISENSE__
#include <cub/device/device_segmented_reduce.cuh>
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html
class DeviceSegmentedReduce : public CubWrapper<DeviceSegmentedReduce>
{
  public:
    DeviceSegmentedReduce(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    // DeviceVector:

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename ReductionOp, typename T>
    DeviceSegmentedReduce& Reduce(DeviceVector<std::byte>& external_buffer,
                                  InputIteratorT           d_in,
                                  OutputIteratorT          d_out,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets,
                                  ReductionOp              reduction_op,
                                  T                        initial_value)
    {


        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_in,
                                                                 d_out,
                                                                 num_segments,
                                                                 d_begin_offsets,
                                                                 d_end_offsets,
                                                                 reduction_op,
                                                                 initial_value,
                                                                 _stream,
                                                                 false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& Sum(DeviceVector<std::byte>& external_buffer,
                               InputIteratorT           d_in,
                               OutputIteratorT          d_out,
                               int                      num_segments,
                               BeginOffsetIteratorT     d_begin_offsets,
                               EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::Sum(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& Min(DeviceVector<std::byte>& external_buffer,
                               InputIteratorT           d_in,
                               OutputIteratorT          d_out,
                               int                      num_segments,
                               BeginOffsetIteratorT     d_begin_offsets,
                               EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::Min(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& ArgMin(DeviceVector<std::byte>& external_buffer,
                                  InputIteratorT           d_in,
                                  OutputIteratorT          d_out,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::ArgMin(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& Max(DeviceVector<std::byte>& external_buffer,
                               InputIteratorT           d_in,
                               OutputIteratorT          d_out,
                               int                      num_segments,
                               BeginOffsetIteratorT     d_begin_offsets,
                               EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::Max(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& ArgMax(DeviceVector<std::byte>& external_buffer,
                                  InputIteratorT           d_in,
                                  OutputIteratorT          d_out,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::ArgMax(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    // DeviceBuffer:

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename ReductionOp, typename T>
    DeviceSegmentedReduce& Reduce(DeviceBuffer<std::byte>& external_buffer,
                                  InputIteratorT           d_in,
                                  OutputIteratorT          d_out,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets,
                                  ReductionOp              reduction_op,
                                  T                        initial_value)
    {


        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_in,
                                                                 d_out,
                                                                 num_segments,
                                                                 d_begin_offsets,
                                                                 d_end_offsets,
                                                                 reduction_op,
                                                                 initial_value,
                                                                 _stream,
                                                                 false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& Sum(DeviceBuffer<std::byte>& external_buffer,
                               InputIteratorT           d_in,
                               OutputIteratorT          d_out,
                               int                      num_segments,
                               BeginOffsetIteratorT     d_begin_offsets,
                               EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::Sum(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& Min(DeviceBuffer<std::byte>& external_buffer,
                               InputIteratorT           d_in,
                               OutputIteratorT          d_out,
                               int                      num_segments,
                               BeginOffsetIteratorT     d_begin_offsets,
                               EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::Min(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& ArgMin(DeviceBuffer<std::byte>& external_buffer,
                                  InputIteratorT           d_in,
                                  OutputIteratorT          d_out,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::ArgMin(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& Max(DeviceBuffer<std::byte>& external_buffer,
                               InputIteratorT           d_in,
                               OutputIteratorT          d_out,
                               int                      num_segments,
                               BeginOffsetIteratorT     d_begin_offsets,
                               EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::Max(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& ArgMax(DeviceBuffer<std::byte>& external_buffer,
                                  InputIteratorT           d_in,
                                  OutputIteratorT          d_out,
                                  int                      num_segments,
                                  BeginOffsetIteratorT     d_begin_offsets,
                                  EndOffsetIteratorT       d_end_offsets)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceSegmentedReduce::ArgMax(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    // Origin:

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename ReductionOp, typename T>
    DeviceSegmentedReduce& Reduce(void*                d_temp_storage,
                                  size_t&              temp_storage_bytes,
                                  InputIteratorT       d_in,
                                  OutputIteratorT      d_out,
                                  int                  num_segments,
                                  BeginOffsetIteratorT d_begin_offsets,
                                  EndOffsetIteratorT   d_end_offsets,
                                  ReductionOp          reduction_op,
                                  T                    initial_value)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(
            cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                               temp_storage_bytes,
                                               d_in,
                                               d_out,
                                               num_segments,
                                               d_begin_offsets,
                                               d_end_offsets,
                                               reduction_op,
                                               initial_value,
                                               _stream,
                                               false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& Sum(void*                d_temp_storage,
                               size_t&              temp_storage_bytes,
                               InputIteratorT       d_in,
                               OutputIteratorT      d_out,
                               int                  num_segments,
                               BeginOffsetIteratorT d_begin_offsets,
                               EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::DeviceSegmentedReduce::Sum(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& Min(void*                d_temp_storage,
                               size_t&              temp_storage_bytes,
                               InputIteratorT       d_in,
                               OutputIteratorT      d_out,
                               int                  num_segments,
                               BeginOffsetIteratorT d_begin_offsets,
                               EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::DeviceSegmentedReduce::Min(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& ArgMin(void*                d_temp_storage,
                                  size_t&              temp_storage_bytes,
                                  InputIteratorT       d_in,
                                  OutputIteratorT      d_out,
                                  int                  num_segments,
                                  BeginOffsetIteratorT d_begin_offsets,
                                  EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::DeviceSegmentedReduce::ArgMin(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& Max(void*                d_temp_storage,
                               size_t&              temp_storage_bytes,
                               InputIteratorT       d_in,
                               OutputIteratorT      d_out,
                               int                  num_segments,
                               BeginOffsetIteratorT d_begin_offsets,
                               EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::DeviceSegmentedReduce::Max(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }


    template <typename InputIteratorT, typename OutputIteratorT>
    DeviceSegmentedReduce& ArgMax(void*           d_temp_storage,
                                  size_t&         temp_storage_bytes,
                                  InputIteratorT  d_in,
                                  OutputIteratorT d_out,
                                  int             num_items)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::DeviceSegmentedReduce::ArgMax(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, _stream, false));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    DeviceSegmentedReduce& ArgMax(void*                d_temp_storage,
                                  size_t&              temp_storage_bytes,
                                  InputIteratorT       d_in,
                                  OutputIteratorT      d_out,
                                  int                  num_segments,
                                  BeginOffsetIteratorT d_begin_offsets,
                                  EndOffsetIteratorT   d_end_offsets)
    {
        MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(cub::DeviceSegmentedReduce::ArgMax(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, _stream, false));
    }
};
}  // namespace muda

#include "details/cub_wrapper_macro_undef.inl"