#pragma once
#include <cuda.h>

namespace muda
{
template <typename T>
class DeviceBuffer;

template <typename T>
class DeviceBuffer2D;

template <typename T>
class DeviceBuffer3D;

class NDReshaper
{
  public:
    //template <typename T, typename FConstruct>
    //static MUDA_HOST void resize(int              grid_dim,
    //                             int              block_dim,
    //                             cudaStream_t     stream,
    //                             DeviceBuffer<T>& buffer,
    //                             size_t           new_size,
    //                             FConstruct&&     fct);

    //template <typename T, typename FConstruct>
    using T = float;
    using FConstruct = std::function<void(Buffer2DView<T>)>;
    static MUDA_HOST void resize(int                grid_dim,
                                 int                block_dim,
                                 cudaStream_t       stream,
                                 DeviceBuffer2D<T>& buffer,
                                 Extent2D           new_extent,
                                 FConstruct&&       fct);
};


}  // namespace muda

#include "details/nd_reshaper.inl"