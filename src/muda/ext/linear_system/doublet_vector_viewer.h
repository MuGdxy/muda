#pragma once

#include <string>
#include <muda/viewer/viewer_base.h>
#include <Eigen/Core>

namespace muda
{
template <bool IsConst, typename T, int N>
class DoubletVectorViewerBase : public ViewerBase<IsConst>
{
    using Base = ViewerBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::auto_const_t<U>;

  public:
    using SegmentVector  = Eigen::Matrix<T, N, 1>;
    using ConstViewer    = DoubletVectorViewerBase<true, T, N>;
    using NonConstViewer = DoubletVectorViewerBase<false, T, N>;
    using ThisViewer     = DoubletVectorViewerBase<IsConst, T, N>;


    struct CDoublet
    {
        MUDA_GENERIC CDoublet(int index, const SegmentVector& segment)
            : index(index)
            , segment_value(segment)
        {
        }
        int                  index;
        const SegmentVector& segment_value;
    };

  protected:
    auto_const_t<int>*           m_segment_indices;
    auto_const_t<SegmentVector>* m_segment_values;

    int m_size                 = 0;
    int m_doublet_index_offset = 0;
    int m_doublet_count        = 0;
    int m_total_doublet_count  = 0;

  public:
    MUDA_GENERIC DoubletVectorViewerBase() = default;
    MUDA_GENERIC DoubletVectorViewerBase(int segment_count,
                                         int doublet_index_offset,
                                         int doublet_count,
                                         int total_doublet_count,
                                         auto_const_t<int>* segment_indices,
                                         auto_const_t<SegmentVector>* segment_values)
        : m_size(segment_count)
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

    // implicit conversion

    MUDA_GENERIC ConstViewer as_const() const noexcept
    {
        return ConstViewer{m_size,
                           m_doublet_index_offset,
                           m_doublet_count,
                           m_total_doublet_count,
                           m_segment_indices,
                           m_segment_values};
    }

    MUDA_GENERIC operator ConstViewer() const noexcept { return as_const(); }

    // non-const access

    MUDA_GENERIC ThisViewer subview(int offset, int count) noexcept
    {
        return ThisViewer{m_size,
                          m_doublet_index_offset + offset,
                          count,
                          m_total_doublet_count,
                          m_segment_indices,
                          m_segment_values};
    }
    MUDA_GENERIC ThisViewer subview(int offset) noexcept
    {
        MUDA_KERNEL_ASSERT(offset < m_doublet_count,
                           "DoubletVectorViewer [%s:%s]: offset is out of range, size=%d, your offset=%d",
                           name(),
                           kernel_name(),
                           m_doublet_count,
                           offset);
        return subview(offset, m_doublet_count - offset);
    }

    MUDA_GENERIC int doublet_count() const noexcept { return m_doublet_count; }
    MUDA_GENERIC int total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }

    MUDA_GENERIC CDoublet operator()(int i) const
    {
        auto index = get_index(i);
        return CDoublet{m_segment_indices[index], m_segment_values[index]};
    }

    MUDA_GENERIC ConstViewer subview(int offset, int count) const
    {
        return remove_const(*this).subview(offset, count);
    }


    MUDA_GENERIC ConstViewer subview(int offset) const
    {
        return remove_const(*this).subview(offset);
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
class CDoubletVectorViewer : public DoubletVectorViewerBase<true, T, N>
{
    using Base = DoubletVectorViewerBase<true, T, N>;
    MUDA_VIEWER_COMMON_NAME(CDoubletVectorViewer);

  public:
    using Base::Base;
    using SegmentVector = typename Base::SegmentVector;
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
class DoubletVectorViewer : public DoubletVectorViewerBase<false, T, N>
{
    using Base = DoubletVectorViewerBase<false, T, N>;
    MUDA_VIEWER_COMMON_NAME(DoubletVectorViewer);

  public:
    using SegmentVector = typename Base::SegmentVector;
    using CDoublet      = typename Base::CDoublet;
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
            MUDA_KERNEL_ASSERT(segment_index >= 0 && segment_index < m_viewer.m_size,
                               "DoubletVectorViewer [%s:%s]: segment_index out of range, m_size=%d, yours=%d",
                               m_viewer.name(),
                               m_viewer.kernel_name(),
                               m_viewer.m_size,
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

    MUDA_GENERIC CDoubletVectorViewer<T, N> subview(int offset, int count) const
    {
        return CDoubletVectorViewer{Base::subview(offset, count)};
    }

    MUDA_GENERIC CDoubletVectorViewer<T, N> subview(int offset) const
    {
        return CDoubletVectorViewer{Base::subview(offset)};
    }

    MUDA_GENERIC DoubletVectorViewer<T, N> subview(int offset, int count)
    {
        return DoubletVectorViewer{Base::subview(offset, count)};
    }

    MUDA_GENERIC DoubletVectorViewer<T, N> subview(int offset)
    {
        return DoubletVectorViewer{Base::subview(offset)};
    }
};

template <bool IsConst, typename T>
class DoubletVectorViewerBase<IsConst, T, 1> : public ViewerBase<IsConst>
{
  public:
    using ConstViewer = DoubletVectorViewerBase<true, T, 1>;
    using Viewer      = DoubletVectorViewerBase<false, T, 1>;
    using ThisViewer  = DoubletVectorViewerBase<IsConst, T, 1>;


    struct CDoublet
    {
        MUDA_GENERIC CDoublet(int index, const T& segment)
            : index(index)
            , value(segment)
        {
        }
        int      index;
        const T& value;
    };

  protected:
    auto_const_t<int>* m_indices;
    auto_const_t<T>*   m_values;

    int m_size                 = 0;
    int m_doublet_index_offset = 0;
    int m_doublet_count        = 0;
    int m_total_doublet_count  = 0;

  public:
    MUDA_GENERIC DoubletVectorViewerBase() = default;
    MUDA_GENERIC DoubletVectorViewerBase(int                size,
                                         int                index_offset,
                                         int                doublet_count,
                                         int                total_doublet_count,
                                         auto_const_t<int>* indices,
                                         auto_const_t<T>*   values)
        : m_size(size)
        , m_doublet_index_offset(index_offset)
        , m_doublet_count(doublet_count)
        , m_total_doublet_count(total_doublet_count)
        , m_indices(indices)
        , m_values(values)
    {
        MUDA_KERNEL_ASSERT(index_offset + doublet_count <= total_doublet_count,
                           "DoubletVectorViewer [%s:%s]: out of range, m_total_doublet_count=%d, "
                           "your doublet_index_offset=%d, doublet_count=%d",
                           name(),
                           kernel_name(),
                           m_total_doublet_count,
                           index_offset,
                           doublet_count);
    }

    // implicit conversion

    MUDA_GENERIC ConstViewer as_const() const noexcept
    {
        return ConstViewer{
            m_size, m_doublet_index_offset, m_doublet_count, m_total_doublet_count, m_indices, m_values};
    }

    MUDA_GENERIC operator ConstViewer() const noexcept { return as_const(); }

    // non-const access

    MUDA_GENERIC ThisViewer subview(int offset, int count) noexcept
    {
        return ThisViewer{
            m_size, m_doublet_index_offset + offset, count, m_total_doublet_count, m_indices, m_values};
    }
    MUDA_GENERIC ThisViewer subview(int offset) noexcept
    {
        MUDA_KERNEL_ASSERT(offset < m_doublet_count,
                           "DoubletVectorViewer [%s:%s]: offset is out of range, size=%d, your offset=%d",
                           name(),
                           kernel_name(),
                           m_doublet_count,
                           offset);
        return subview(offset, m_doublet_count - offset);
    }

    MUDA_GENERIC int doublet_count() const noexcept { return m_doublet_count; }
    MUDA_GENERIC int total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }

    MUDA_GENERIC CDoublet operator()(int i) const
    {
        auto index = get_index(i);
        return CDoublet{m_indices[index], m_values[index]};
    }

    MUDA_GENERIC ConstViewer subview(int offset, int count) const
    {
        return remove_const(*this).subview(offset, count);
    }


    MUDA_GENERIC ConstViewer subview(int offset) const
    {
        return remove_const(*this).subview(offset);
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

template <typename T>
class CDoubletVectorViewer<T, 1> : public DoubletVectorViewerBase<true, T, 1>
{
    using Base = DoubletVectorViewerBase<true, T, 1>;
    MUDA_VIEWER_COMMON_NAME(CDoubletVectorViewer);

  public:
    using Base::Base;
    using ThisViewer = CDoubletVectorViewer<T, 1>;
    MUDA_GENERIC CDoubletVectorViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC ThisViewer subview(int offset, int count) const
    {
        return ThisViewer{Base::subview(offset, count)};
    }

    MUDA_GENERIC ThisViewer subview(int offset) const
    {
        return ThisViewer{Base::subview(offset)};
    }
};

template <typename T>
class DoubletVectorViewer<T, 1> : public DoubletVectorViewerBase<false, T, 1>
{
    using Base = DoubletVectorViewerBase<false, T, 1>;
    MUDA_VIEWER_COMMON_NAME(DoubletVectorViewer);

  public:
    using CDoublet    = typename Base::CDoublet;
    using ThisViewer  = DoubletVectorViewer<T, 1>;
    using ConstViewer = CDoubletVectorViewer<T, 1>;
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
            return CDoublet{m_viewer.m_indices[m_index], m_viewer.m_values[m_index]};
        }

        MUDA_GENERIC void write(int index, const T& value) &&
        {
            MUDA_KERNEL_ASSERT(index >= 0 && index < m_viewer.m_size,
                               "DoubletVectorViewer [%s:%s]: segment_index out of range, m_size=%d, yours=%d",
                               m_viewer.name(),
                               m_viewer.kernel_name(),
                               m_viewer.m_size,
                               index);

            m_viewer.m_indices[m_index] = index;
            m_viewer.m_values[m_index]  = value;
        }

        MUDA_GENERIC ~Proxy() = default;
    };

    using Base::operator();

    MUDA_GENERIC Proxy operator()(int i)
    {
        auto index = Base::get_index(i);
        return Proxy{*this, index};
    }

    MUDA_GENERIC ConstViewer subview(int offset, int count) const
    {
        return ConstViewer{Base::subview(offset, count)};
    }

    MUDA_GENERIC ConstViewer subview(int offset) const
    {
        return ConstViewer{Base::subview(offset)};
    }

    MUDA_GENERIC ThisViewer subview(int offset, int count)
    {
        return ThisViewer{Base::subview(offset, count)};
    }

    MUDA_GENERIC ThisViewer subview(int offset)
    {
        return ThisViewer{Base::subview(offset)};
    }
};
}  // namespace muda

#include "details/doublet_vector_viewer.inl"
