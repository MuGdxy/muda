#pragma once
#include <Eigen/Core>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/viewer/viewer_base.h>
#include <cublas_v2.h>
namespace muda
{
template <bool IsConst, typename T>
class DenseVectorViewerBase : public ViewerBase<IsConst>
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "now only support real number");

  public:
    using CBufferView    = CBufferView<T>;
    using BufferView     = BufferView<T>;
    using ThisBufferView = std::conditional_t<IsConst, CBufferView, BufferView>;

    using ConstViewer    = DenseVectorViewerBase<true, T>;
    using NonConstViewer = DenseVectorViewerBase<false, T>;
    using ThisViewer = std::conditional_t<IsConst, ConstViewer, NonConstViewer>;

    using VectorType = Eigen::Vector<T, Eigen::Dynamic>;
    template <typename T>
    using MapVectorT =
        Eigen::Map<T, Eigen::AlignmentType::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using MapVector     = MapVectorT<VectorType>;
    using CMapVector    = MapVectorT<const VectorType>;
    using ThisMapVector = std::conditional_t<IsConst, CMapVector, MapVector>;

  protected:
    ThisBufferView m_view;
    size_t         m_offset = 0;
    size_t         m_size;

  public:
    MUDA_GENERIC DenseVectorViewerBase(ThisBufferView view, size_t offset, size_t size)
        : m_view(view)
        , m_offset(offset)
        , m_size(size)
    {
    }

    MUDA_GENERIC auto as_const() const
    {
        return ConstViewer{m_view, m_offset, m_size};
    }

    operator ConstViewer() const { return as_const(); }


    MUDA_GENERIC auto segment(size_t offset, size_t size) const
    {
        check_segment(offset, size);
        auto ret             = ConstViewer{m_view, m_offset + offset, size};
        auto acc             = muda::details::ViewerBaseAccessor();
        acc.kernel_name(ret) = acc.kernel_name(*this);
        acc.viewer_name(ret) = acc.viewer_name(*this);
        return ret;
    }

    MUDA_GENERIC auto segment(size_t offset, size_t size)
    {
        check_segment(offset, size);
        auto ret             = ThisViewer{m_view, m_offset + offset, size};
        auto acc             = muda::details::ViewerBaseAccessor();
        acc.kernel_name(ret) = acc.kernel_name(*this);
        acc.viewer_name(ret) = acc.viewer_name(*this);
        return ret;
    }

    MUDA_GENERIC Eigen::VectorBlock<CMapVector> as_eigen() const
    {
        check_data();
        return CMapVector{m_view.origin_data(),
                          (int)origin_size(),
                          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>{1, 1}}
            .segment(m_offset, m_size);
    }

    MUDA_GENERIC operator Eigen::VectorBlock<CMapVector>() const
    {
        return as_eigen();
    }

    MUDA_GENERIC Eigen::VectorBlock<ThisMapVector> as_eigen()
    {
        check_data();
        return ThisMapVector{m_view.origin_data(),
                             (int)origin_size(),
                             Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>{1, 1}}
            .segment(m_offset, m_size);
    }

    MUDA_GENERIC operator Eigen::VectorBlock<ThisMapVector>()
    {
        return as_eigen();
    }


    MUDA_GENERIC const T& operator()(size_t i) const
    {
        return *m_view.data(index(i));
    }
    MUDA_GENERIC auto_const_t<T>& operator()(size_t i)
    {
        return *m_view.data(index(i));
    }

    MUDA_GENERIC auto           size() const { return m_size; }
    MUDA_GENERIC auto           offset() const { return m_offset; }
    MUDA_GENERIC auto           origin_size() const { return m_view.size(); }
    MUDA_GENERIC CBufferView    buffer_view() const { return m_view; }
    MUDA_GENERIC ThisBufferView buffer_view() { return m_view; }

  private:
    MUDA_INLINE MUDA_GENERIC void check_size_matching(int N)
    {
        MUDA_KERNEL_ASSERT(m_size == N,
                           "DenseVectorViewerBase [%s:%s]: size not match, yours size=%lld, expected size=%d",
                           name(),
                           kernel_name(),
                           m_size,
                           N);
    }

    MUDA_INLINE MUDA_GENERIC size_t index(size_t i) const
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
        return m_offset + i;
    }

    MUDA_INLINE MUDA_GENERIC void check_data() const
    {
        MUDA_KERNEL_ASSERT(m_view.data(),
                           "DenseVectorViewerBase [%s:%s]: data is null",
                           name(),
                           kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_segment(size_t offset, size_t size) const
    {
        MUDA_KERNEL_ASSERT(offset + size <= m_size,
                           "DenseVectorViewerBase [%s:%s]: segment out of range, m_size=%lld, offset=%lld, size=%lld",
                           name(),
                           kernel_name(),
                           m_size,
                           offset,
                           size);
    }
};

template <typename T>
using CDenseVectorViewer = DenseVectorViewerBase<true, T>;
template <typename T>
using DenseVectorViewer = DenseVectorViewerBase<false, T>;
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