#include <muda/viewer/viewer_base_accessor.h>
#include <muda/atomic.h>

namespace muda
{
template <typename T>
MUDA_GENERIC auto DenseVectorViewerBase<T>::as_eigen() const
    -> Eigen::VectorBlock<CMapVector>
{
    MUDA_KERNEL_ASSERT(m_view.data(),
                       "DenseVectorViewerBase [%s:%s]: data is null",
                       name(),
                       kernel_name());
    return CMapVector{m_view.origin_data(),
                      (int)origin_size(),
                      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>{1, 1}}
        .segment(m_offset, m_size);
}

template <typename T>
MUDA_GENERIC DenseVectorViewerBase<T> DenseVectorViewerBase<T>::segment(size_t offset,
                                                                        size_t size) const
{
    MUDA_KERNEL_ASSERT(offset + size <= m_size,
                       "DenseVectorViewerBase [%s:%s]: segment out of range, m_size=%lld, offset=%lld, size=%lld",
                       name(),
                       kernel_name(),
                       m_size,
                       offset,
                       size);

    auto ret = DenseVectorViewerBase{m_view, m_offset + offset, size};
    auto acc = muda::details::ViewerBaseAccessor();
    acc.kernel_name(ret) = acc.kernel_name(*this);
    acc.viewer_name(ret) = acc.viewer_name(*this);
    return ret;
}
template <typename T>
MUDA_GENERIC const T& DenseVectorViewerBase<T>::operator()(size_t i) const
{
    MUDA_KERNEL_ASSERT(m_view.data(),
                       "DenseVectorViewerBase [%s:%s]: data is null",
                       name(),
                       kernel_name());
    MUDA_KERNEL_ASSERT(i < m_size,
                       "DenseVectorViewerBase [%s:%s]: index out of range, size=%lld, yours index=%lld",
                       name(),
                       kernel_name(),
                       m_size,
                       i);

    i += m_offset;
    return *m_view.data(i);
}

template <typename T>
MUDA_GENERIC DenseVectorViewerBase<T>::operator Eigen::VectorBlock<CMapVector>() const
{
    return as_eigen();
}

/**************************************************************************
*
*                           CDenseVectorViewer
* 
**************************************************************************/

template <typename T>
MUDA_GENERIC CDenseVectorViewer<T> CDenseVectorViewer<T>::segment(size_t offset, size_t size) const
{
    return Base::segment(offset, size);
}

template <typename T>
template <size_t N>
MUDA_GENERIC CDenseVectorViewer<T> CDenseVectorViewer<T>::segment(size_t offset) const
{
    return MUDA_GENERIC CDenseVectorViewer();
}
/**************************************************************************
*
*                           DenseVectorViewer
* 
**************************************************************************/

template <typename T>
MUDA_GENERIC DenseVectorViewer<T>::operator Eigen::VectorBlock<MapVector>()
{
    return as_eigen();
}

template <typename T>
MUDA_GENERIC DenseVectorViewer<T> DenseVectorViewer<T>::segment(size_t offset, size_t size)
{
    return Base::segment(offset, size);
}

template <typename T>
MUDA_GENERIC CDenseVectorViewer<T> DenseVectorViewer<T>::segment(size_t offset, size_t size) const
{
    return CDenseVectorViewer{*this}.segment(offset, size);
}

template <typename T>
template <size_t N>
MUDA_GENERIC DenseVectorViewer<T> DenseVectorViewer<T>::segment(size_t offset)
{
    return segment(offset, N);
}

template <typename T>
template <size_t N>
MUDA_GENERIC CDenseVectorViewer<T> DenseVectorViewer<T>::segment(size_t offset) const
{
    return segment(offset, N);
}


template <typename T>
MUDA_DEVICE T DenseVectorViewer<T>::atomic_add(size_t i, T val)
{
    auto ptr = &operator()(i);
    return muda::atomic_add(ptr, val);
}

template <typename T>
template <int N>
MUDA_DEVICE Eigen::Vector<T, N> DenseVectorViewer<T>::atomic_add(const Eigen::Vector<T, N>& other)
{
    check_size_matching(N);
    Eigen::Vector<T, N> ret;
#pragma unroll
    for(int i = 0; i < N; ++i)
        ret(i) = atomic_add(i, other(i));
    return ret;
}

template <typename T>
template <size_t N>
MUDA_GENERIC DenseVectorViewer<T>& DenseVectorViewer<T>::operator=(const Eigen::Vector<T, N>& other)
{
    check_size_matching(N);
#pragma unroll
    for(size_t i = 0; i < N; ++i)
        (*this)(i) = other(i);
    return *this;
}

template <typename T>
MUDA_GENERIC auto DenseVectorViewer<T>::as_eigen() -> Eigen::VectorBlock<MapVector>
{
    MUDA_KERNEL_ASSERT(
        m_view.data(), "DenseVectorViewer [%s:%s]: data is null", name(), kernel_name());

    return MapVector(const_cast<T*>(m_view.origin_data()),
                     (int)origin_size(),
                     Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>{1, 1})
        .segment(m_offset, m_size);
}

template <typename T>
MUDA_GENERIC T& DenseVectorViewer<T>::operator()(size_t i)
{
    return const_cast<T&>(Base::operator()(i));
}
template <typename T>
MUDA_GENERIC void DenseVectorViewer<T>::check_size_matching(int N)
{
    MUDA_KERNEL_ASSERT(m_size == N,
                       "DenseVectorViewer [%s:%s]: segment size mismatching, Viewer=(%lld), yours=(%lld)",
                       name(),
                       kernel_name(),
                       m_size,
                       N);
}
}  // namespace muda
