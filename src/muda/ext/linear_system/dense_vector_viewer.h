#pragma once
#include <Eigen/Core>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/viewer/viewer_base.h>
#include <cublas_v2.h>
namespace muda
{
template <typename T>
class DenseVectorViewerBase : public ViewerBase
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "now only support real number");

  protected:
    BufferView<T> m_view;
    size_t        m_offset = 0;
    size_t        m_size;

  public:
    using VectorType = Eigen::Vector<T, Eigen::Dynamic>;
    template <typename T>
    using MapVectorT =
        Eigen::Map<T, Eigen::AlignmentType::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using MapVector  = MapVectorT<VectorType>;
    using CMapVector = MapVectorT<const VectorType>;

    MUDA_GENERIC DenseVectorViewerBase(const BufferView<T>& view, size_t offset, size_t size)
        : m_view(view)
        , m_offset(offset)
        , m_size(size)
    {
    }

    MUDA_GENERIC DenseVectorViewerBase<T> segment(size_t offset, size_t size) const;


    MUDA_GENERIC operator Eigen::VectorBlock<CMapVector>() const;

    MUDA_GENERIC Eigen::VectorBlock<CMapVector> as_eigen() const;

    MUDA_GENERIC const T& operator()(size_t i) const;

    MUDA_GENERIC auto size() const { return m_size; }
    MUDA_GENERIC auto offset() const { return m_offset; }
    MUDA_GENERIC auto origin_size() const { return m_view.size(); }
    MUDA_GENERIC auto buffer_view() const { return m_view; }
};

template <typename T>
class CDenseVectorViewer : public DenseVectorViewerBase<T>
{
    MUDA_VIEWER_COMMON_NAME(CDenseVectorViewer);
    using Base       = DenseVectorViewerBase<T>;
    using CMapVector = typename Base::CMapVector;

  public:
    MUDA_GENERIC CDenseVectorViewer(const CBufferView<T>& view, size_t offset, size_t size)
        : Base(BufferViewBase<T>{view}, offset, size)
    {
    }

    MUDA_GENERIC CDenseVectorViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CDenseVectorViewer<T> segment(size_t offset, size_t size) const;

    template <size_t N>
    MUDA_GENERIC CDenseVectorViewer<T> segment(size_t offset) const;
};

template <typename T>
class DenseVectorViewer : public DenseVectorViewerBase<T>
{
    MUDA_VIEWER_COMMON_NAME(DenseVectorViewer);

    using Base       = DenseVectorViewerBase<T>;
    using MapVector  = typename Base::MapVector;
    using CMapVector = typename Base::CMapVector;

  public:
    using Base::Base;
    MUDA_GENERIC DenseVectorViewer(Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC DenseVectorViewer(CDenseVectorViewer<T>&) = delete;

    MUDA_GENERIC DenseVectorViewer<T> segment(size_t offset, size_t size);

    MUDA_GENERIC CDenseVectorViewer<T> segment(size_t offset, size_t) const;

    template <size_t N>
    MUDA_GENERIC DenseVectorViewer<T> segment(size_t offset);

    template <size_t N>
    MUDA_GENERIC CDenseVectorViewer<T> segment(size_t offset) const;

    MUDA_DEVICE T atomic_add(size_t i, T val);

    template <int N>
    MUDA_DEVICE Eigen::Vector<T, N> atomic_add(const Eigen::Vector<T, N>& other);

    template <size_t N>
    MUDA_GENERIC DenseVectorViewer<T>& operator=(const Eigen::Vector<T, N>& other);

    MUDA_GENERIC operator Eigen::VectorBlock<MapVector>();

    using Base::as_eigen;
    MUDA_GENERIC Eigen::VectorBlock<MapVector> as_eigen();

    MUDA_GENERIC T& operator()(size_t i);

  private:
    MUDA_GENERIC void check_size_matching(int N);
};
}  // namespace muda

namespace muda
{
template <typename T>
struct read_only_viewer<DenseVectorViewer<T>>
{
    using type = CDenseVectorViewer<T>;
};

template <typename T>
struct read_write_viewer<CDenseVectorViewer<T>>
{
    using type = DenseVectorViewer<T>;
};
}  // namespace muda

#include "details/dense_vector_viewer.inl"