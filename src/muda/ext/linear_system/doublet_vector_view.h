#pragma once
#include <muda/view/view_base.h>
#include <muda/ext/linear_system/doublet_vector_viewer.h>

namespace muda
{
template <bool IsConst, typename T, int N>
class DoubletVectorViewBase : public ViewBase<IsConst>
{
  public:
    using SegmentVector = Eigen::Matrix<T, N, 1>;
    using ConstView     = DoubletVectorViewBase<true, T, N>;
    using NonConstView  = DoubletVectorViewBase<false, T, N>;
    using ThisView      = DoubletVectorViewBase<IsConst, T, N>;

    using CViewer    = CDoubletVectorViewer<T, N>;
    using Viewer     = DoubletVectorViewer<T, N>;
    using ThisViewer = std::conditional_t<IsConst, CViewer, Viewer>;

  protected:
    auto_const_t<int>*           m_segment_indices;
    auto_const_t<SegmentVector>* m_segment_values;

    int m_segment_count        = 0;
    int m_doublet_index_offset = 0;
    int m_doublet_count        = 0;
    int m_total_doublet_count  = 0;

  public:
    MUDA_GENERIC DoubletVectorViewBase() = default;
    MUDA_GENERIC DoubletVectorViewBase(int                segment_count,
                                       int                doublet_index_offset,
                                       int                doublet_count,
                                       int                total_doublet_count,
                                       auto_const_t<int>* segment_indices,
                                       auto_const_t<SegmentVector>* segment_values)
        : m_segment_count(segment_count)
        , m_doublet_index_offset(doublet_index_offset)
        , m_doublet_count(doublet_count)
        , m_total_doublet_count(total_doublet_count)
        , m_segment_indices(segment_indices)
        , m_segment_values(segment_values)
    {
        MUDA_KERNEL_ASSERT(doublet_index_offset + doublet_count <= total_doublet_count,
                           "DoubletVectorView: out of range, m_total_doublet_count=%d, "
                           "your doublet_index_offset=%d, doublet_count=%d",
                           m_total_doublet_count,
                           doublet_index_offset,
                           doublet_count);
    }

    // implicit conversion

    MUDA_GENERIC ConstView as_const() const noexcept
    {
        return ConstView{m_segment_count,
                         m_doublet_index_offset,
                         m_doublet_count,
                         m_total_doublet_count,
                         m_segment_indices,
                         m_segment_values};
    }

    MUDA_GENERIC operator ConstView() const noexcept { return as_const(); }

    // non-const access

    MUDA_GENERIC ThisView subview(int offset, int count) noexcept
    {
        return ThisView{m_segment_count,
                        m_doublet_index_offset + offset,
                        count,
                        m_total_doublet_count,
                        m_segment_indices,
                        m_segment_values};
    }
    MUDA_GENERIC ThisView subview(int offset) noexcept
    {
        MUDA_KERNEL_ASSERT(offset < m_doublet_count,
                           "DoubletVectorView : offset is out of range, size=%d, your offset=%d",
                           m_doublet_count,
                           offset);
        return subview(offset, m_doublet_count - offset);
    }

    MUDA_GENERIC int doublet_count() const noexcept { return m_doublet_count; }
    MUDA_GENERIC int total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }

    MUDA_GENERIC ConstView subview(int offset, int count) const
    {
        return remove_const(*this).subview(offset, count);
    }

    MUDA_GENERIC ConstView subview(int offset) const
    {
        return remove_const(*this).subview(offset);
    }

    MUDA_GENERIC ThisViewer viewer() noexcept
    {
        return ThisViewer{m_segment_count,
                          m_doublet_index_offset,
                          m_doublet_count,
                          m_total_doublet_count,
                          m_segment_indices,
                          m_segment_values};
    }

    MUDA_GENERIC CViewer cviewer() const noexcept
    {
        return CViewer{m_segment_count,
                       m_doublet_index_offset,
                       m_doublet_count,
                       m_total_doublet_count,
                       m_segment_indices,
                       m_segment_values};
    }
};

template <bool IsConst, typename T>
class DoubletVectorViewBase<IsConst, T, 1> : public ViewBase<IsConst>
{
  public:
    using ConstView    = DoubletVectorViewBase<true, T, 1>;
    using NonConstView = DoubletVectorViewBase<false, T, 1>;
    using ThisView     = DoubletVectorViewBase<IsConst, T, 1>;

    using CViewer    = CDoubletVectorViewer<T, 1>;
    using Viewer     = DoubletVectorViewer<T, 1>;
    using ThisViewer = std::conditional_t<IsConst, CViewer, Viewer>;

  protected:
    auto_const_t<int>* m_indices;
    auto_const_t<T>*   m_values;

    int m_size                 = 0;
    int m_doublet_index_offset = 0;
    int m_doublet_count        = 0;
    int m_total_doublet_count  = 0;

  public:
    MUDA_GENERIC DoubletVectorViewBase() = default;
    MUDA_GENERIC DoubletVectorViewBase(int                count,
                                       int                doublet_index_offset,
                                       int                doublet_count,
                                       int                total_doublet_count,
                                       auto_const_t<int>* indices,
                                       auto_const_t<T>*   values)
        : m_size(count)
        , m_doublet_index_offset(doublet_index_offset)
        , m_doublet_count(doublet_count)
        , m_total_doublet_count(total_doublet_count)
        , m_indices(indices)
        , m_values(values)
    {
        MUDA_KERNEL_ASSERT(doublet_index_offset + doublet_count <= total_doublet_count,
                           "DoubletVectorView: out of range, m_total_doublet_count=%d, "
                           "your doublet_index_offset=%d, doublet_count=%d",
                           m_total_doublet_count,
                           doublet_index_offset,
                           doublet_count);
    }

    // implicit conversion

    MUDA_GENERIC ConstView as_const() const noexcept
    {
        return ConstView{m_size, m_doublet_index_offset, m_doublet_count, m_total_doublet_count, m_indices, m_values};
    }

    MUDA_GENERIC operator ConstView() const noexcept { return as_const(); }

    // non-const access

    MUDA_GENERIC ThisView subview(int offset, int count) noexcept
    {
        return ThisView{m_size, m_doublet_index_offset + offset, count, m_total_doublet_count, m_indices, m_values};
    }
    MUDA_GENERIC ThisView subview(int offset) noexcept
    {
        MUDA_KERNEL_ASSERT(offset < m_doublet_count,
                           "DoubletVectorView : offset is out of range, size=%d, your offset=%d",
                           m_doublet_count,
                           offset);
        return subview(offset, m_doublet_count - offset);
    }

    MUDA_GENERIC int doublet_count() const noexcept { return m_doublet_count; }
    MUDA_GENERIC int total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }

    MUDA_GENERIC ConstView subview(int offset, int count) const
    {
        return remove_const(*this).subview(offset, count);
    }

    MUDA_GENERIC ConstView subview(int offset) const
    {
        return remove_const(*this).subview(offset);
    }

    MUDA_GENERIC ThisViewer viewer() noexcept
    {
        return ThisViewer{
            m_size, m_doublet_index_offset, m_doublet_count, m_total_doublet_count, m_indices, m_values};
    }

    MUDA_GENERIC CViewer cviewer() const noexcept
    {
        return CViewer{m_size, m_doublet_index_offset, m_doublet_count, m_total_doublet_count, m_indices, m_values};
    }
};

template <typename T, int N>
using DoubletVectorView = DoubletVectorViewBase<false, T, N>;
template <typename T, int N>
using CDoubletVectorView = DoubletVectorViewBase<true, T, N>;
}  // namespace muda

namespace muda
{
template <typename Ty, int N>
struct read_only_viewer<DoubletVectorView<Ty, N>>
{
    using type = CDoubletVectorView<Ty, N>;
};

template <typename Ty, int N>
struct read_write_viewer<CDoubletVectorView<Ty, N>>
{
    using type = DoubletVectorView<Ty, N>;
};
}  // namespace muda

#include "details/doublet_vector_view.inl"
