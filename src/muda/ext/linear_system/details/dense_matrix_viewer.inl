#include <muda/viewer/viewer_base_accessor.h>
#include <muda/atomic.h>

namespace muda
{
template <bool IsConst, typename T>
MUDA_GENERIC auto DenseMatrixViewerBase<IsConst, T>::block(size_t row_offset,
                                                           size_t col_offset,
                                                           size_t row_size,
                                                           size_t col_size) -> ThisViewer
{
    MUDA_KERNEL_ASSERT(row_offset + row_size <= m_row_size && col_offset + col_size <= m_col_size,
                       "DenseMatrixViewerBase [%s:%s]: block index out of range, shape=(%lld,%lld), yours index=(%lld,%lld)",
                       this->name(),
                       this->kernel_name(),
                       m_row_size,
                       m_col_size,
                       row_offset,
                       col_offset);

    auto ret = DenseMatrixViewerBase{
        m_view, m_row_offset + row_offset, m_col_offset + col_offset, row_size, col_size};
    ret.copy_name(*this);
    return ret;
}

template <bool IsConst, typename T>
MUDA_GENERIC auto DenseMatrixViewerBase<IsConst, T>::as_eigen()
    -> Eigen::Block<ThisMapMatrix>
{
    auto outer = m_view.pitch_bytes() / sizeof(T);

    return ThisMapMatrix{m_view.origin_data(),
                         (int)origin_row(),
                         (int)origin_col(),
                         Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>{(int)outer, 1}}
        .block(m_row_offset, m_col_offset, m_row_size, m_col_size);
}

template <bool IsConst, typename T>
MUDA_GENERIC auto DenseMatrixViewerBase<IsConst, T>::operator()(size_t i, size_t j)
    -> auto_const_t<T>&
{
    if constexpr(DEBUG_VIEWER)
    {
        MUDA_KERNEL_ASSERT(m_view.data(0),
                           "DenseMatrixViewer [%s:%s]: data is null",
                           this->name(),
                           this->kernel_name());
        if(m_row_offset == 0 && m_col_offset == 0)
        {
            MUDA_KERNEL_ASSERT(i < m_row_size && j < m_col_size,
                               "DenseMatrixViewer [%s:%s]: index out of range, shape=(%lld,%lld), yours index=(%lld,%lld)",
                               this->name(),
                               this->kernel_name(),
                               m_row_size,
                               m_col_size,
                               i,
                               j);
        }
        else
        {
            MUDA_KERNEL_ASSERT(i < m_row_size && j < m_col_size,
                               "DenseMatrixViewer [%s:%s]:index out of range, block shape=(%lld,%lld), your index=(%lld,%lld)",
                               this->name(),
                               this->kernel_name(),
                               m_row_size,
                               m_col_size,
                               i,
                               j);
        }
    }
    i += m_row_offset;
    j += m_col_offset;
    return *m_view.data(j, i);
}

template <bool IsConst, typename T>
MUDA_GENERIC auto DenseMatrixViewerBase<IsConst, T>::as_eigen() const
    -> Eigen::Block<CMapMatrix>
{
    auto outer = m_view.pitch_bytes() / sizeof(T);

    return CMapMatrix{m_view.origin_data(),
                      (int)origin_row(),
                      (int)origin_col(),
                      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>{(int)outer, 1}}
        .block(m_row_offset, m_col_offset, m_row_size, m_col_size);
}

template <bool IsConst, typename T>
MUDA_GENERIC size_t DenseMatrixViewerBase<IsConst, T>::origin_row() const
{
    size_t ret;
    ret = m_view.extent().width();
    return ret;
}

template <bool IsConst, typename T>
MUDA_GENERIC size_t DenseMatrixViewerBase<IsConst, T>::origin_col() const
{
    size_t ret;
    ret = m_view.extent().height();
    return ret;
}

/**************************************************************************
* 
*                           DenseMatrixViewer
* 
**************************************************************************/

//template <typename T>
//MUDA_GENERIC DenseMatrixViewer<T> DenseMatrixViewer<T>::block(size_t row_offset,
//                                                              size_t col_offset,
//                                                              size_t row_size,
//                                                              size_t col_size) const
//{
//    return Base::block(row_offset, col_offset, row_size, col_size);
//}
//
//template <typename T>
//template <size_t M, size_t N>
//MUDA_GENERIC DenseMatrixViewer<T> DenseMatrixViewer<T>::block(size_t row_offset,
//                                                              size_t col_offset) const
//{
//    return Base::block<M, N>(row_offset, col_offset);
//}

template <typename T>
template <int M, int N>
MUDA_DEVICE Eigen::Matrix<T, M, N> DenseMatrixViewer<T>::atomic_add(const Eigen::Matrix<T, M, N>& other)
{
    check_size_matching(M, N);
    Eigen::Matrix<T, M, N> ret;
#pragma unroll
    for(int i = 0; i < M; ++i)
#pragma unroll
        for(int j = 0; j < N; ++j)
        {
            ret(i, j) = atomic_add(i, j, other(i, j));
        }
    return ret;
}

template <typename T>
template <int M, int N>
MUDA_GENERIC DenseMatrixViewer<T>& DenseMatrixViewer<T>::operator=(const Eigen::Matrix<T, M, N>& other)
{
    check_size_matching(M, N);
#pragma unroll
    for(int i = 0; i < M; ++i)
#pragma unroll
        for(int j = 0; j < N; ++j)
            (*this)(i, j) = other(i, j);
    return *this;
}

template <typename T>
MUDA_DEVICE T DenseMatrixViewer<T>::atomic_add(size_t i, size_t j, T val)
{
    auto ptr = &this->operator()(i, j);
    muda::atomic_add(ptr, val);
    return val;
}

template <typename T>
MUDA_INLINE MUDA_GENERIC void DenseMatrixViewer<T>::check_size_matching(int M, int N) const
{
    MUDA_KERNEL_ASSERT(this->m_row_size == M && this->m_col_size == N,
                       "DenseMatrixViewer [%s:%s] shape mismatching, Viewer=(%lld,%lld), yours=(%lld,%lld)",
                       this->name(),
                       this->kernel_name(),
                       this->m_row_size,
                       this->m_col_size,
                       M,
                       N);
}
}  // namespace muda
