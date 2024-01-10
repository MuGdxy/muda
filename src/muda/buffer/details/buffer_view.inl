#include <muda/buffer/buffer_launch.h>
#include <muda/compute_graph/compute_graph_builder.h>

namespace muda
{
template <bool IsConst, typename T>
MUDA_GENERIC auto BufferViewBase<IsConst, T>::subview(size_t offset, size_t size) const
    MUDA_NOEXCEPT->ConstView
{
    return remove_const(*this).subview(offset, size);
}

template <bool IsConst, typename T>
MUDA_GENERIC auto BufferViewBase<IsConst, T>::subview(size_t offset,
                                                      size_t size) MUDA_NOEXCEPT->ThisView
{
#ifndef __CUDA_ARCH__
    if(ComputeGraphBuilder::is_topo_building())
        return ThisView{};  // dummy
#endif

    if(size == ~0)
        size = m_size - offset;
    MUDA_KERNEL_ASSERT(offset + size <= m_size,
                       "BufferView out of range, size = %d, yours = %d",
                       m_size,
                       offset + size);
    return ThisView{m_data, m_offset + offset, size};
}

template <bool IsConst, typename T>
MUDA_GENERIC auto BufferViewBase<IsConst, T>::viewer() MUDA_NOEXCEPT->ThisViewer
{
    return ThisViewer{data(), static_cast<int>(m_size)};
}

template <bool IsConst, typename T>
MUDA_GENERIC auto BufferViewBase<IsConst, T>::cviewer() const MUDA_NOEXCEPT->CViewer
{
    return CViewer{data(), static_cast<int>(m_size)};
}

template <typename T>
MUDA_HOST void BufferView<T>::fill(const T& v)
{
    BufferLaunch()
        .fill(*this, v)  //
        .wait();
}

template <typename T>
MUDA_HOST void BufferView<T>::copy_from(CBufferView<T> other)
{
    BufferLaunch()
        .copy(*this, other)  //
        .wait();
}

template <typename T>
MUDA_HOST void BufferView<T>::copy_from(const T* host)
{
    BufferLaunch()
        .copy(*this, host)  //
        .wait();
}

template <typename T>
MUDA_HOST void CBufferView<T>::copy_to(T* host) const
{
    BufferLaunch()
        .copy(host, *this)  //
        .wait();
}
}  // namespace muda