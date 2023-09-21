#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_histogram.cuh>
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_histogram.html

class DeviceHistogram : public CubWrapper<DeviceHistogram>
{
  public:
    DeviceHistogram(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    // HistogramEven (single channel, 1D input)
    template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    DeviceHistogram& HistogramEven(DeviceBuffer<std::byte>& external_buffer,
                                   SampleIteratorT           d_samples,
                                   CounterT*                 d_histogram,
                                   int                       num_levels,
                                   LevelT                    lower_level,
                                   LevelT                    upper_level,
                                   OffsetT                   num_samples)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_samples,
                                                                  d_histogram,
                                                                  num_levels,
                                                                  lower_level,
                                                                  upper_level,
                                                                  num_samples,
                                                                  m_stream,
                                                                  false));
    }

    // HistogramEven (single channel, 2D input)
    template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    DeviceHistogram& HistogramEven(DeviceBuffer<std::byte>& external_buffer,
                                   SampleIteratorT           d_samples,
                                   CounterT*                 d_histogram,
                                   int                       num_levels,
                                   LevelT                    lower_level,
                                   LevelT                    upper_level,
                                   OffsetT                   num_row_samples,
                                   OffsetT                   num_rows,
                                   size_t                    row_stride_bytes)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_samples,
                                                                  d_histogram,
                                                                  num_levels,
                                                                  lower_level,
                                                                  upper_level,
                                                                  num_row_samples,
                                                                  num_rows,
                                                                  row_stride_bytes,
                                                                  m_stream,
                                                                  false));
    }

    // MultiHistogramEven (multiple channels, 1D input)
    template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    DeviceHistogram& MultiHistogramEven(DeviceBuffer<std::byte>& external_buffer,
                                        SampleIteratorT d_samples,
                                        CounterT* d_histogram[NUM_ACTIVE_CHANNELS],
                                        int    num_levels[NUM_ACTIVE_CHANNELS],
                                        LevelT lower_level[NUM_ACTIVE_CHANNELS],
                                        LevelT upper_level[NUM_ACTIVE_CHANNELS],
                                        OffsetT num_pixels)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceHistogram::MultiHistogramEven(
            d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, lower_level, upper_level, num_pixels, m_stream, false));
    }

    // MultiHistogramEven (multiple channels, 2D input)
    template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    DeviceHistogram& MultiHistogramEven(DeviceBuffer<std::byte>& external_buffer,
                                        SampleIteratorT d_samples,
                                        CounterT* d_histogram[NUM_ACTIVE_CHANNELS],
                                        int    num_levels[NUM_ACTIVE_CHANNELS],
                                        LevelT lower_level[NUM_ACTIVE_CHANNELS],
                                        LevelT upper_level[NUM_ACTIVE_CHANNELS],
                                        OffsetT num_row_pixels,
                                        OffsetT num_rows,
                                        size_t  row_stride_bytes)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceHistogram::MultiHistogramEven(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histogram,
            num_levels,
            lower_level,
            upper_level,
            num_row_pixels,
            num_rows,
            row_stride_bytes,
            m_stream,
            false));
    }

    // HistogramRange (single channel, 1D input)
    template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    DeviceHistogram& HistogramRange(DeviceBuffer<std::byte>& external_buffer,
                                    SampleIteratorT           d_samples,
                                    CounterT*                 d_histogram,
                                    int                       num_levels,
                                    LevelT*                   d_levels,
                                    OffsetT                   num_samples)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceHistogram::HistogramRange(
            d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_samples, m_stream, false));
    }

    // HistogramRange (single channel, 2D input)
    template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    DeviceHistogram& HistogramRange(DeviceBuffer<std::byte>& external_buffer,
                                    SampleIteratorT           d_samples,
                                    CounterT*                 d_histogram,
                                    int                       num_levels,
                                    LevelT*                   d_levels,
                                    OffsetT                   num_row_samples,
                                    OffsetT                   num_rows,
                                    size_t                    row_stride_bytes)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceHistogram::HistogramRange(d_temp_storage,
                                                                   temp_storage_bytes,
                                                                   d_samples,
                                                                   d_histogram,
                                                                   num_levels,
                                                                   d_levels,
                                                                   num_row_samples,
                                                                   num_rows,
                                                                   row_stride_bytes,
                                                                   m_stream,
                                                                   false));
    }

    // MultiHistogramRange (multiple channels, 1D input)
    template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    DeviceHistogram& MultiHistogramRange(DeviceBuffer<std::byte>& external_buffer,
                                         SampleIteratorT d_samples,
                                         CounterT* d_histogram[NUM_ACTIVE_CHANNELS],
                                         int num_levels[NUM_ACTIVE_CHANNELS],
                                         LevelT* d_levels[NUM_ACTIVE_CHANNELS],
                                         OffsetT num_pixels)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceHistogram::MultiHistogramRange(
            d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels, d_levels, num_pixels, m_stream, false));
    }

    // MultiHistogramRange (multiple channels, 2D input)
    template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    DeviceHistogram& MultiHistogramRange(DeviceBuffer<std::byte>& external_buffer,
                                         SampleIteratorT d_samples,
                                         CounterT* d_histogram[NUM_ACTIVE_CHANNELS],
                                         int num_levels[NUM_ACTIVE_CHANNELS],
                                         LevelT* d_levels[NUM_ACTIVE_CHANNELS],
                                         OffsetT num_row_pixels,
                                         OffsetT num_rows,
                                         size_t  row_stride_bytes)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceHistogram::MultiHistogramRange(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_histogram,
            num_levels,
            d_levels,
            num_row_pixels,
            num_rows,
            row_stride_bytes,
            m_stream,
            false));
    }
};
}  // namespace muda
