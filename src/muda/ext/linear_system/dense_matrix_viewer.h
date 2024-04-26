#pragma once
#include <Eigen/Core>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/viewer/viewer_base.h>
#include <cublas_v2.h>
namespace muda
{
template <bool IsConst, typename T>
class DenseMatrixViewerBase : public ViewerBase<IsConst>
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "now only support real number");
    static_assert(!std::is_const_v<T>, "T must be non-const type");

    using Base = ViewerBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    using CBuffer2DView = CBuffer2DView<T>;
    using Buffer2DView  = Buffer2DView<T>;
    using ThisBuffer2DView = std::conditional_t<IsConst, CBuffer2DView, Buffer2DView>;

    using ConstViewer    = DenseMatrixViewerBase<true, T>;
    using NonConstViewer = DenseMatrixViewerBase<false, T>;
    using ThisViewer = std::conditional_t<IsConst, ConstViewer, NonConstViewer>;

    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    template <typename U>
    using MapMatrixT =
        Eigen::Map<U, Eigen::AlignmentType::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using MapMatrix     = MapMatrixT<MatrixType>;
    using CMapMatrix    = MapMatrixT<const MatrixType>;
    using ThisMapMatrix = std::conditional_t<IsConst, CMapMatrix, MapMatrix>;

  protected:
    ThisBuffer2DView m_view;
    size_t           m_row_offset = 0;
    size_t           m_col_offset = 0;
    size_t           m_row_size   = 0;
    size_t           m_col_size   = 0;

  public:
    MUDA_GENERIC DenseMatrixViewerBase(ThisBuffer2DView view,
                                       size_t           row_offset,
                                       size_t           col_offset,
                                       size_t           row_size,
                                       size_t           col_size)
        : m_view(view)
        , m_row_offset(row_offset)
        , m_col_offset(col_offset)
        , m_row_size(row_size)
        , m_col_size(col_size)
    {
    }

    // implicit conversion

    MUDA_GENERIC auto as_const() const
    {
        return ConstViewer{m_view, m_row_offset, m_col_offset, m_row_size, m_col_size};
    }

    MUDA_GENERIC operator ConstViewer() const { return as_const(); }

    // non-const accessor

    MUDA_GENERIC ThisViewer block(size_t row_offset, size_t col_offset, size_t row_size, size_t col_size);
    template <int M, int N>
    MUDA_GENERIC ThisViewer block(int row_offset, int col_offset)
    {
        return block(row_offset, col_offset, M, N);
    }
    MUDA_GENERIC Eigen::Block<ThisMapMatrix> as_eigen();
    MUDA_GENERIC operator Eigen::Block<CMapMatrix>();
    MUDA_GENERIC auto_const_t<T>& operator()(size_t i, size_t j);
    MUDA_GENERIC auto             buffer_view() { return m_view; }

    // const accessor

    MUDA_GENERIC ConstViewer block(size_t row_offset, size_t col_offset, size_t row_size, size_t col_size) const
    {
        return remove_const(*this).block(row_offset, col_offset, row_size, col_size);
    }
    template <int M, int N>
    MUDA_GENERIC ConstViewer block(int row_offset, int col_offset) const
    {
        return remove_const(*this).block<M, N>(row_offset, col_offset);
    }
    MUDA_GENERIC Eigen::Block<CMapMatrix> as_eigen() const;
    MUDA_GENERIC operator Eigen::Block<CMapMatrix>() const
    {
        return as_eigen();
    }
    MUDA_GENERIC const T& operator()(size_t i, size_t j) const
    {
        return remove_const(*this)(i, j);
    }

    MUDA_GENERIC size_t row() const { return m_row_size; }
    MUDA_GENERIC size_t col() const { return m_col_size; }
    MUDA_GENERIC size_t origin_row() const;
    MUDA_GENERIC size_t origin_col() const;
    MUDA_GENERIC auto   buffer_view() const { return m_view; }
    MUDA_GENERIC auto   row_offset() const { return m_row_offset; }
    MUDA_GENERIC auto   col_offset() const { return m_col_offset; }
};

template <typename T>
class CDenseMatrixViewer : public DenseMatrixViewerBase<true, T>
{
    MUDA_VIEWER_COMMON_NAME(CDenseMatrixViewer);

    using Base       = DenseMatrixViewerBase<true, T>;
    using CMapMatrix = typename Base::CMapMatrix;

  public:
    using Base::Base;

    MUDA_GENERIC CDenseMatrixViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CDenseMatrixViewer block(size_t row_offset,
                                          size_t col_offset,
                                          size_t row_size,
                                          size_t col_size) const
    {
        return Base::block(row_offset, col_offset, row_size, col_size);
    }

    template <size_t M, size_t N>
    MUDA_GENERIC CDenseMatrixViewer block(size_t row_offset, size_t col_offset) const
    {
        return Base::template block<M, N>(row_offset, col_offset);
    }
};

template <typename T>
class DenseMatrixViewer : public DenseMatrixViewerBase<false, T>
{
    MUDA_VIEWER_COMMON_NAME(DenseMatrixViewer);

    using Base       = DenseMatrixViewerBase<false, T>;
    using MapMatrix  = typename Base::MapMatrix;
    using CMapMatrix = typename Base::CMapMatrix;

  public:
    using Base::Base;

    MUDA_GENERIC DenseMatrixViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC DenseMatrixViewer(const CDenseMatrixViewer<T>&) = delete;

    MUDA_GENERIC DenseMatrixViewer block(size_t row_offset, size_t col_offset, size_t row_size, size_t col_size)
    {
        return Base::block(row_offset, col_offset, row_size, col_size);
    }

    template <size_t M, size_t N>
    MUDA_GENERIC DenseMatrixViewer block(size_t row_offset, size_t col_offset)
    {
        return Base::template block<M, N>(row_offset, col_offset);
    }

    MUDA_DEVICE T atomic_add(size_t i, size_t j, T val);

    template <int M, int N>
    MUDA_DEVICE Eigen::Matrix<T, M, N> atomic_add(const Eigen::Matrix<T, M, N>& other);

    template <int M, int N>
    MUDA_GENERIC DenseMatrixViewer& operator=(const Eigen::Matrix<T, M, N>& other);

  private:
    MUDA_GENERIC void check_size_matching(int M, int N) const;
};

}  // namespace muda

//namespace muda
//{
//template <typename T>
//struct read_only_viewer<DenseMatrixViewer<T>>
//{
//    using type = CDenseMatrixViewer<T>;
//};
//
//template <typename T>
//struct read_write_viewer<CDenseMatrixViewer<T>>
//{
//    using type = DenseMatrixViewer<T>;
//};
//}  // namespace muda

#include "details/dense_matrix_viewer.inl"
