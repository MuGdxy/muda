#pragma once

#include <string>
#include <muda/viewer/viewer_base.h>
#include <muda/buffer/device_buffer.h>
#include <Eigen/Core>

namespace muda
{
template <typename T, int N>
class DoubletVectorViewerBase : public muda::ViewerBase
{
  public:
    using SegmentVector = Eigen::Matrix<T, N, 1>;

    struct CDoublet
    {
        MUDA_GENERIC CDoublet(int index, const SegmentVector& segment)
            : segment_index(index)
            , segment_value(segment)
        {
        }
        int                  segment_index;
        const SegmentVector& segment_value;
    };


  protected:
    int*           m_segment_indices;
    SegmentVector* m_segment_values;

    int m_segment_count = 0;

    int m_doublet_index_offset = 0;
    int m_doublet_count        = 0;
    int m_total_doublet_count  = 0;

  public:
    MUDA_GENERIC DoubletVectorViewerBase() = default;
    MUDA_GENERIC DoubletVectorViewerBase(int            segment_count,
                                         int            doublet_index_offset,
                                         int            doublet_count,
                                         int            total_doublet_count,
                                         int*           segment_indices,
                                         SegmentVector* segment_values)
        : m_segment_count(segment_count)
        , m_doublet_index_offset(doublet_index_offset)
        , m_doublet_count(doublet_count)
        , m_total_doublet_count(total_doublet_count)
        , m_segment_indices(segment_indices)
        , m_segment_values(segment_values)
    {
        MUDA_KERNEL_ASSERT(doublet_index_offset + doublet_count <= total_doublet_count,
                           "DoubletVectorViewer [%s:%s]: out of range, m_total_doublet_count=%d, "
                           "your doublet_index_offset=%d, doublet_count=%d",
                           name(),
                           kernel_name(),
                           m_total_doublet_count,
                           doublet_index_offset,
                           doublet_count);
    }

    MUDA_GENERIC int doublet_count() const noexcept { return m_doublet_count; }
    MUDA_GENERIC int total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }

    MUDA_GENERIC void write(int i, int segment_index, const SegmentVector& segment)
    {
        MUDA_ASSERT(segment_index >= 0 && segment_index < m_segment_count,
                    "DoubletVectorViewer [%s:%s]: segment_index out of range, m_segment_count=%d, yours=%d",
                    name(),
                    kernel_name(),
                    m_segment_count,
                    segment_index);

        auto index               = get_index(i);
        m_segment_indices[index] = segment_index;
        m_segment_values[index]  = segment;
    }

    MUDA_GENERIC CDoublet operator()(int i) const
    {
        auto index = get_index(i);
        return CDoublet{m_segment_indices[index], m_segment_values[index]};
    }

    MUDA_GENERIC DoubletVectorViewerBase<T, N> subview(int offset, int count) const
    {
        return DoubletVectorViewerBase<T, N>{m_segment_count,
                                             m_doublet_index_offset + offset,
                                             count,
                                             m_total_doublet_count,
                                             m_segment_indices,
                                             m_segment_values};
    }

    MUDA_GENERIC DoubletVectorViewerBase<T, N> subview(int offset) const
    {
        MUDA_KERNEL_ASSERT(offset < m_doublet_count,
                           "DoubletVectorViewer [%s:%s]: offset is out of range, size=%d, your offset=%d",
                           name(),
                           kernel_name(),
                           m_doublet_count,
                           offset);
        return subview(offset, m_doublet_count - offset);
    }

  protected:
    MUDA_INLINE MUDA_GENERIC int get_index(int i) const noexcept
    {

        MUDA_KERNEL_ASSERT(i >= 0 && i < m_doublet_count,
                           "DoubletVectorViewer [%s:%s]: index out of range, m_doublet_count=%d, your index=%d",
                           kernel_name(),
                           name(),
                           m_doublet_count,
                           i);
        auto index = i + m_doublet_index_offset;
        return index;
    }
};

template <typename T, int N>
class CDoubletVectorViewer : public DoubletVectorViewerBase<T, N>
{
    using Base = DoubletVectorViewerBase<T, N>;
    MUDA_VIEWER_COMMON_NAME(CDoubletVectorViewer);

  public:
    using Base::Base;
    using SegmentVector = typename Base::SegmentVector;
    MUDA_GENERIC CDoubletVectorViewer(int                  segment_count,
                                      int                  doublet_index_offset,
                                      int                  doublet_count,
                                      int                  total_doublet_count,
                                      const int*           segment_indices,
                                      const SegmentVector* segment_values)
        : Base(segment_count,
               doublet_index_offset,
               doublet_count,
               total_doublet_count,
               const_cast<int*>(segment_indices),
               const_cast<SegmentVector*>(segment_values))
    {
    }

    MUDA_GENERIC CDoubletVectorViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CDoubletVectorViewer<T, N> subview(int offset, int count) const
    {
        return CDoubletVectorViewer{Base::subview(offset, count)};
    }

    MUDA_GENERIC CDoubletVectorViewer<T, N> subview(int offset) const
    {
        return CDoubletVectorViewer{Base::subview(offset)};
    }
};

template <typename T, int N>
class DoubletVectorViewer : public DoubletVectorViewerBase<T, N>
{
    using Base = DoubletVectorViewerBase<T, N>;
    MUDA_VIEWER_COMMON_NAME(DoubletVectorViewer);

  public:
    using CDoublet = typename Base::CDoublet;
    using Base::Base;
    MUDA_GENERIC DoubletVectorViewer(const Base& base)
        : Base(base)
    {
    }

    class Proxy
    {
        friend class DoubletVectorViewer;
        DoubletVectorViewer& m_viewer;
        int                  m_index = 0;

      private:
        MUDA_GENERIC Proxy(DoubletVectorViewer& viewer, int index)
            : m_viewer(viewer)
            , m_index(index)
        {
        }

      public:
        MUDA_GENERIC auto read() &&
        {
            return CDoublet{m_viewer.m_segment_indices[m_index],
                            m_viewer.m_segment_values[m_index]};
        }

        MUDA_GENERIC void write(int segment_index, const SegmentVector& block) &&
        {
            MUDA_KERNEL_ASSERT(segment_index >= 0 && segment_index < m_viewer.m_segment_count,
                               "DoubletVectorViewer [%s:%s]: segment_index out of range, m_segment_count=%d, yours=%d",
                               m_viewer.name(),
                               m_viewer.kernel_name(),
                               m_viewer.m_segment_count,
                               segment_index);

            m_viewer.m_segment_indices[m_index] = segment_index;
            m_viewer.m_segment_values[m_index]  = block;
        }

        MUDA_GENERIC ~Proxy() = default;
    };

    using Base::operator();

    MUDA_GENERIC Proxy operator()(int i)
    {
        auto index = Base::get_index(i);
        return Proxy{*this, index};
    }

    MUDA_GENERIC DoubletVectorViewer<T, N> subview(int offset, int count) const
    {
        return DoubletVectorViewer{Base::subview(offset, count)};
    }

    MUDA_GENERIC DoubletVectorViewer<T, N> subview(int offset) const
    {
        return DoubletVectorViewer{Base::subview(offset)};
    }
};
}  // namespace muda

#include "details/doublet_vector_viewer.inl"
