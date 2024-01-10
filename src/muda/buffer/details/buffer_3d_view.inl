#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/buffer/buffer_launch.h>

namespace muda
{
template <bool IsConst, typename T>
MUDA_GENERIC auto Buffer3DViewBase<IsConst, T>::data(size_t x, size_t y, size_t z)
    MUDA_NOEXCEPT->auto_const_t<T>*
{
    x += m_offset.offset_in_depth();
    y += m_offset.offset_in_height();
    z += m_offset.offset_in_width();
    auto depth_begin =
        reinterpret_cast<std::byte*>(remove_const(m_data)) + m_pitch_bytes_area * x;
    auto height_begin = depth_begin + m_pitch_bytes * y;
    return reinterpret_cast<T*>(height_begin) + z;
}
template <bool IsConst, typename T>
MUDA_GENERIC auto Buffer3DViewBase<IsConst, T>::data(size_t flatten_i)
    MUDA_NOEXCEPT->auto_const_t<T>*
{
    auto area       = m_extent.width() * m_extent.height();
    auto x          = flatten_i / area;
    auto i_in_area  = flatten_i % area;
    auto y          = i_in_area / m_extent.width();
    auto i_in_width = i_in_area % m_extent.width();
    auto z          = i_in_width;
    return data(x, y, z);
}

template <bool IsConst, typename T>
MUDA_GENERIC size_t Buffer3DViewBase<IsConst, T>::total_size() const MUDA_NOEXCEPT
{
    return m_extent.width() * m_extent.height() * m_extent.depth();
}

template <bool IsConst, typename T>
MUDA_GENERIC auto Buffer3DViewBase<IsConst, T>::subview(Offset3D offset, Extent3D extent)
    MUDA_NOEXCEPT->ThisView
{
#ifndef __CUDA_ARCH__
    if(ComputeGraphBuilder::is_topo_building())
        return ThisView{};  // dummy
#endif

    if(!extent.valid())
        extent = m_extent - as_extent(offset);

    MUDA_KERNEL_ASSERT(extent + as_extent(offset) <= m_extent,
                       "Buffer3DView out of range, extent = (%d,%d,%d), yours = (%d,%d,%d)",
                       (int)m_extent.depth(),
                       (int)m_extent.height(),
                       (int)m_extent.width(),
                       (int)extent.depth(),
                       (int)extent.height(),
                       (int)extent.width());
    return ThisView{m_data, m_pitch_bytes, m_pitch_bytes_area, m_origin_width, m_origin_height, offset, extent};
}

template <bool IsConst, typename T>
MUDA_GENERIC auto Buffer3DViewBase<IsConst, T>::viewer() MUDA_NOEXCEPT->ThisViewer
{
    return ThisViewer{m_data,
                      make_int3(m_offset.offset_in_depth(),
                                m_offset.offset_in_height(),
                                m_offset.offset_in_width()),
                      make_int3(m_extent.depth(), m_extent.height(), m_extent.width()),
                      (int)m_pitch_bytes,
                      (int)m_pitch_bytes_area};
}

template <bool IsConst, typename T>
MUDA_GENERIC auto Buffer3DViewBase<IsConst, T>::cviewer() const MUDA_NOEXCEPT->CViewer
{
    return remove_const(*this).viewer();
}

template <typename T>
MUDA_HOST void CBuffer3DView<T>::copy_to(T* host) const
{
    BufferLaunch().copy(host, *this).wait();
}

template <typename T>
MUDA_HOST void Buffer3DView<T>::fill(const T& v)
{
    BufferLaunch().fill(*this, v).wait();
}

template <typename T>
MUDA_HOST void Buffer3DView<T>::copy_from(const Buffer3DView<T>& other)
{
    BufferLaunch().copy(*this, other).wait();
}

template <typename T>
MUDA_HOST void Buffer3DView<T>::copy_from(const T* host)
{
    BufferLaunch().copy(*this, host).wait();
}
}  // namespace muda