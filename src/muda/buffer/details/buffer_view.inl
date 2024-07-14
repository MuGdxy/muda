#include <muda/buffer/buffer_launch.h>
#include <muda/compute_graph/compute_graph_builder.h>

namespace muda
{
template <bool IsConst, typename T>
MUDA_GENERIC auto BufferViewT<IsConst, T>::subview(size_t offset, size_t size) const
    MUDA_NOEXCEPT->ThisView
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
MUDA_GENERIC auto BufferViewT<IsConst, T>::viewer() const MUDA_NOEXCEPT->ThisViewer
{
    return ThisViewer{data(), static_cast<int>(m_size)};
}

template <bool IsConst, typename T>
MUDA_GENERIC auto BufferViewT<IsConst, T>::cviewer() const MUDA_NOEXCEPT->CViewer
{
    return CViewer{data(), static_cast<int>(m_size)};
}

template <bool IsConst, typename T>
MUDA_HOST void BufferViewT<IsConst, T>::fill(const T& v) const MUDA_REQUIRES(!IsConst)
{
    static_assert(!IsConst, "This must be non-const");

    BufferLaunch()
        .fill(*this, v)  //
        .wait();
}

template <bool IsConst, typename T>
MUDA_HOST void BufferViewT<IsConst, T>::copy_from(const BufferViewT<true, T>& other) const
    MUDA_REQUIRES(!IsConst)
{
    static_assert(!IsConst, "This must be non-const");

    BufferLaunch()
        .template copy<T>(*this, other)  //
        .wait();
}

template <bool IsConst, typename T>
MUDA_HOST void BufferViewT<IsConst, T>::copy_from(const T* host) const
    MUDA_REQUIRES(!IsConst)
{
    static_assert(!IsConst, "This must be non-const");

    BufferLaunch()
        .copy(host, *this)  //
        .wait();
}

template <bool IsConst, typename T>
MUDA_HOST void BufferViewT<IsConst, T>::copy_to(T* host) const
{
    BufferLaunch()
        .copy(host, *this)  //
        .wait();
}
}  // namespace muda