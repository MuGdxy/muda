#pragma once
#include <Eigen/Core>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/viewer/viewer_base.h>
#include <cublas_v2.h>
namespace muda
{
template <typename T>
class DenseMatrixViewerBase : public ViewerBase
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "now only support real number");

  protected:
    Buffer2DView<T> m_view;
    size_t          m_row_offset = 0;
    size_t          m_col_offset = 0;
    size_t          m_row_size   = 0;
    size_t          m_col_size   = 0;

  public:
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    template <typename T>
    using MapMatrixT =
        Eigen::Map<T, Eigen::AlignmentType::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using MapMatrix  = MapMatrixT<MatrixType>;
    using CMapMatrix = MapMatrixT<const MatrixType>;

    MUDA_GENERIC DenseMatrixViewerBase(const Buffer2DView<T>& view,
                                       size_t                 row_offset,
                                       size_t                 col_offset,
                                       size_t                 row_size,
                                       size_t                 col_size)
        : m_view(view)
        , m_row_offset(row_offset)
        , m_col_offset(col_offset)
        , m_row_size(row_size)
        , m_col_size(col_size)
    {
    }

    MUDA_GENERIC DenseMatrixViewerBase block(size_t row_offset,
                                             size_t col_offset,
                                             size_t row_size,
                                             size_t col_size) const;

    template <int M, int N>
    MUDA_GENERIC DenseMatrixViewerBase block(int row_offset, int col_offset) const
    {
        return block(row_offset, col_offset, M, N);
    }

    MUDA_GENERIC operator Eigen::Block<CMapMatrix>() const;

    MUDA_GENERIC Eigen::Block<CMapMatrix> as_eigen() const;

    MUDA_GENERIC const T& operator()(size_t i, size_t j) const;

    MUDA_GENERIC size_t row() const { return m_row_size; }
    MUDA_GENERIC size_t col() const { return m_col_size; }
    MUDA_GENERIC size_t origin_row() const;
    MUDA_GENERIC size_t origin_col() const;
    MUDA_GENERIC auto   buffer_view() const { return m_view; }
    MUDA_GENERIC auto   row_offset() const { return m_row_offset; }
    MUDA_GENERIC auto   col_offset() const { return m_col_offset; }
};

template <typename T>
class CDenseMatrixViewer : public DenseMatrixViewerBase<T>
{
    MUDA_VIEWER_COMMON_NAME(CDenseMatrixViewer);

    using Base       = DenseMatrixViewerBase<T>;
    using CMapMatrix = typename Base::CMapMatrix;

  public:
    using Base::Base;

    MUDA_GENERIC CDenseMatrixViewer(const CBuffer2DView<T>& view,
                                    size_t                  row_offset,
                                    size_t                  col_offset,
                                    size_t                  row_size,
                                    size_t                  col_size)
        : Base(Buffer2DViewBase{view}, row_offset, col_offset, row_size, col_size)
    {
    }

    MUDA_GENERIC CDenseMatrixViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CDenseMatrixViewer block(size_t row_offset,
                                          size_t col_offset,
                                          size_t row_size,
                                          size_t col_size) const;

    template <size_t M, size_t N>
    MUDA_GENERIC CDenseMatrixViewer block(size_t row_offset, size_t col_offset) const;
};

template <typename T>
class DenseMatrixViewer : public DenseMatrixViewerBase<T>
{
    MUDA_VIEWER_COMMON_NAME(DenseMatrixViewer);

    using Base       = DenseMatrixViewerBase<T>;
    using MapMatrix  = typename Base::MapMatrix;
    using CMapMatrix = typename Base::CMapMatrix;

  public:
    using Base::Base;

    MUDA_GENERIC DenseMatrixViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC DenseMatrixViewer(const CDenseMatrixViewer<T>&) = delete;

    MUDA_GENERIC DenseMatrixViewer block(size_t row_offset,
                                         size_t col_offset,
                                         size_t row_size,
                                         size_t col_size) const;

    template <size_t M, size_t N>
    MUDA_GENERIC DenseMatrixViewer block(size_t row_offset, size_t col_offset) const;

    MUDA_GENERIC T& operator()(size_t i, size_t j);

    MUDA_DEVICE T atomic_add(size_t i, size_t j, T val);

    template <int M, int N>
    MUDA_DEVICE Eigen::Matrix<T, M, N> atomic_add(const Eigen::Matrix<T, M, N>& other);

    template <int M, int N>
    MUDA_GENERIC DenseMatrixViewer& operator=(const Eigen::Matrix<T, M, N>& other);

    MUDA_GENERIC operator Eigen::Block<MapMatrix>();

    using Base::as_eigen;
    MUDA_GENERIC Eigen::Block<MapMatrix> as_eigen();

  private:
    MUDA_GENERIC void check_size_matching(int M, int N) const;
};

}  // namespace muda

namespace muda
{
template <typename T>
struct read_only_viewer<DenseMatrixViewer<T>>
{
    using type = CDenseMatrixViewer<T>;
};

template <typename T>
struct read_write_viewer<CDenseMatrixViewer<T>>
{
    using type = DenseMatrixViewer<T>;
};
}  // namespace muda

#include "details/dense_matrix_viewer.inl"
