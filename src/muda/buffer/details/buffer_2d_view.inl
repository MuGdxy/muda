#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/buffer/buffer_launch.h>

namespace muda
{
template <typename T>
MUDA_GENERIC const T* Buffer2DViewBase<T>::data(size_t x, size_t y) const MUDA_NOEXCEPT
{
    x += m_offset.offset_in_height();
    y += m_offset.offset_in_width();

    auto height_begin = reinterpret_cast<std::byte*>(m_data) + m_pitch_bytes * x;
    return reinterpret_cast<T*>(height_begin) + y;
}

template <typename T>
MUDA_GENERIC const T* Buffer2DViewBase<T>::data(size_t flatten_i) const MUDA_NOEXCEPT
{
    auto x = flatten_i / m_extent.width();
    auto y = flatten_i % m_extent.width();
    return data(x, y);
}

template <typename T>
MUDA_GENERIC Buffer2DViewBase<T> Buffer2DViewBase<T>::subview(Offset2D offset,
                                                              Extent2D extent) const MUDA_NOEXCEPT
{
#ifndef __CUDA_ARCH__
    if(ComputeGraphBuilder::is_topo_building())
        return Buffer2DViewBase<T>{};  // dummy
#endif

    if(!extent.valid())
        extent = m_extent - as_extent(offset);

    MUDA_KERNEL_ASSERT(extent + as_extent(offset) <= m_extent,
                       "Buffer2DView out of range, extent = (%d,%d), yours = (%d,%d)",
                       (int)m_extent.height(),
                       (int)m_extent.width(),
                       (int)extent.height(),
                       (int)extent.width());
    return Buffer2DViewBase<T>{
        const_cast<T*>(m_data), m_pitch_bytes, m_origin_width, m_origin_height, offset, extent};
}
template <typename T>
MUDA_GENERIC cudaPitchedPtr Buffer2DViewBase<T>::cuda_pitched_ptr() const MUDA_NOEXCEPT
{
    return make_cudaPitchedPtr(m_data, m_pitch_bytes, m_origin_width * sizeof(T), m_origin_height);
}

template <typename T>
MUDA_GENERIC CDense2D<T> Buffer2DViewBase<T>::cviewer() const MUDA_NOEXCEPT
{
    return CDense2D<T>{m_data,
                       make_int2((int)m_offset.offset_in_height(),
                                 (int)m_offset.offset_in_width()),
                       make_int2((int)m_extent.height(), (int)m_extent.width()),
                       (int)m_pitch_bytes};
}

template <typename T>
MUDA_HOST void CBuffer2DView<T>::copy_to(T* host) const
{
    BufferLaunch().copy<T>(host, *this).wait();
}


template <typename T>
MUDA_GENERIC Dense2D<T> Buffer2DView<T>::viewer() const MUDA_NOEXCEPT
{
    return Dense2D<T>{m_data,
                      make_int2((int)m_offset.offset_in_height(),
                                (int)m_offset.offset_in_width()),
                      make_int2((int)m_extent.height(), (int)m_extent.width()),
                      (int)m_pitch_bytes};
}

template <typename T>
MUDA_HOST void Buffer2DView<T>::fill(const T& val)
{
    BufferLaunch().fill(*this, val).wait();
}

template <typename T>
MUDA_HOST void Buffer2DView<T>::copy_from(CBuffer2DView<T> other)
{
    BufferLaunch().copy(*this, other).wait();
}

template <typename T>
MUDA_HOST void Buffer2DView<T>::copy_from(const T* host)
{
    BufferLaunch().copy(*this, host).wait();
}

}  // namespace muda