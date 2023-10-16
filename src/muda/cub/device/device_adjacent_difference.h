#pragma once
#if 0
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/device_adjacent_difference.cuh>
#else
namespace cub
{
class Difference;
}
#endif

namespace muda
{
//ref: https://nvlabs.github.io/cub/structcub_1_1_device_adjacent_difference.html
class DeviceAdjacentDifference : public CubWrapper<DeviceAdjacentDifference>
{
  public:
    DeviceAdjacentDifference(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    // DeviceVector:

    template <typename InputIteratorT, typename OutputIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractLeftCopy(DeviceVector<std::byte>& external_buffer,
                                               InputIteratorT  d_in,
                                               OutputIteratorT d_out,
                                               int             num_items,
                                               DifferenceOpT difference_op = {},
                                               bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceAdjacentDifference::SubtractLeftCopy(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, difference_op, this->stream(), debug_synchronous));
    }

    template <typename RandomAccessIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractLeft(DeviceVector<std::byte>& external_buffer,
                                           RandomAccessIteratorT d_in,
                                           int                   num_items,
                                           DifferenceOpT difference_op = {},
                                           bool debug_synchronous      = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceAdjacentDifference::SubtractLeft(
            d_temp_storage, temp_storage_bytes, d_in, num_items, difference_op, this->stream(), debug_synchronous));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractRightCopy(DeviceVector<std::byte>& external_buffer,
                                                InputIteratorT  d_in,
                                                OutputIteratorT d_out,
                                                int             num_items,
                                                DifferenceOpT difference_op = {},
                                                bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceAdjacentDifference::SubtractRightCopy(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, difference_op, this->stream(), debug_synchronous));
    }

    template <typename RandomAccessIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractRight(DeviceVector<std::byte>& external_buffer,
                                            RandomAccessIteratorT d_in,
                                            int                   num_items,
                                            DifferenceOpT difference_op = {},
                                            bool debug_synchronous      = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceAdjacentDifference::SubtractRight(
            d_temp_storage, temp_storage_bytes, d_in, num_items, difference_op, this->stream(), debug_synchronous));
    }

    // DeviceBuffer:

    
    template <typename InputIteratorT, typename OutputIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractLeftCopy(DeviceBuffer<std::byte>& external_buffer,
                                               InputIteratorT  d_in,
                                               OutputIteratorT d_out,
                                               int             num_items,
                                               DifferenceOpT difference_op = {},
                                               bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceAdjacentDifference::SubtractLeftCopy(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, difference_op, this->stream(), debug_synchronous));
    }

    template <typename RandomAccessIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractLeft(DeviceBuffer<std::byte>& external_buffer,
                                           RandomAccessIteratorT d_in,
                                           int                   num_items,
                                           DifferenceOpT difference_op = {},
                                           bool debug_synchronous      = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceAdjacentDifference::SubtractLeft(
            d_temp_storage, temp_storage_bytes, d_in, num_items, difference_op, this->stream(), debug_synchronous));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractRightCopy(DeviceBuffer<std::byte>& external_buffer,
                                                InputIteratorT  d_in,
                                                OutputIteratorT d_out,
                                                int             num_items,
                                                DifferenceOpT difference_op = {},
                                                bool debug_synchronous = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceAdjacentDifference::SubtractRightCopy(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, difference_op, this->stream(), debug_synchronous));
    }

    template <typename RandomAccessIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractRight(DeviceBuffer<std::byte>& external_buffer,
                                            RandomAccessIteratorT d_in,
                                            int                   num_items,
                                            DifferenceOpT difference_op = {},
                                            bool debug_synchronous      = false)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::DeviceAdjacentDifference::SubtractRight(
            d_temp_storage, temp_storage_bytes, d_in, num_items, difference_op, this->stream(), debug_synchronous));
    }

    // Origin:

    
    template <typename InputIteratorT, typename OutputIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractLeftCopy(void*  d_temp_storage, size_t& temp_storage_bytes,
                                               InputIteratorT  d_in,
                                               OutputIteratorT d_out,
                                               int             num_items,
                                               DifferenceOpT difference_op = {},
                                               bool debug_synchronous = false)
    {
        checkCudaErrors(cub::DeviceAdjacentDifference::SubtractLeftCopy(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, difference_op, this->stream(), debug_synchronous));
    }

    template <typename RandomAccessIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractLeft(void*  d_temp_storage, size_t& temp_storage_bytes,
                                           RandomAccessIteratorT d_in,
                                           int                   num_items,
                                           DifferenceOpT difference_op = {},
                                           bool debug_synchronous      = false)
    {
        checkCudaErrors(cub::DeviceAdjacentDifference::SubtractLeft(
            d_temp_storage, temp_storage_bytes, d_in, num_items, difference_op, this->stream(), debug_synchronous));
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractRightCopy(void*  d_temp_storage, size_t& temp_storage_bytes,
                                                InputIteratorT  d_in,
                                                OutputIteratorT d_out,
                                                int             num_items,
                                                DifferenceOpT difference_op = {},
                                                bool debug_synchronous = false)
    {
        checkCudaErrors(cub::DeviceAdjacentDifference::SubtractRightCopy(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, difference_op, this->stream(), debug_synchronous));
    }

    template <typename RandomAccessIteratorT, typename DifferenceOpT = cub::Difference>
    DeviceAdjacentDifference& SubtractRight(void*  d_temp_storage, size_t& temp_storage_bytes,
                                            RandomAccessIteratorT d_in,
                                            int                   num_items,
                                            DifferenceOpT difference_op = {},
                                            bool debug_synchronous      = false)
    {
        checkCudaErrors(cub::DeviceAdjacentDifference::SubtractRight(
            d_temp_storage, temp_storage_bytes, d_in, num_items, difference_op, this->stream(), debug_synchronous));
    }
};
}  // namespace muda
#endif