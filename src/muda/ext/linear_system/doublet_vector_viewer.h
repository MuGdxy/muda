#pragma once

#include <string>
#include <muda/viewer/viewer_base.h>
#include <Eigen/Core>

/*
* - 2024/2/23 remove viewer's subview, view's subview is enough
*/

namespace muda
{
template <bool IsConst, typename T, int N>
class DoubletVectorViewerBase : public ViewerBase<IsConst>
{
    using Base = ViewerBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

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
    // vector info
    int m_total_segment_count = 0;

    // doublet info
    int m_doublet_index_offset = 0;
    int m_doublet_count        = 0;
    int m_total_doublet_count  = 0;

    // subvector info
    int m_subvector_offset = 0;
    int m_subvector_extent = 0;

    // data
    auto_const_t<int>*           m_segment_indices;
    auto_const_t<SegmentVector>* m_segment_values;

  public:
    MUDA_GENERIC DoubletVectorViewerBase() = default;
    MUDA_GENERIC DoubletVectorViewerBase(int total_segment_count,
                                         int doublet_index_offset,
                                         int doublet_count,
                                         int total_doublet_count,
                                         int subvector_offset,
                                         int subvector_extent,
                                         auto_const_t<int>* segment_indices,
                                         auto_const_t<SegmentVector>* segment_values)
        : m_total_segment_count(total_segment_count)
        , m_doublet_index_offset(doublet_index_offset)
        , m_doublet_count(doublet_count)
        , m_total_doublet_count(total_doublet_count)
        , m_subvector_offset(subvector_offset)
        , m_subvector_extent(subvector_extent)
        , m_segment_indices(segment_indices)
        , m_segment_values(segment_values)
    {
        MUDA_KERNEL_ASSERT(doublet_index_offset + doublet_count <= total_doublet_count,
                           "DoubletVectorViewer: out of range, m_total_doublet_count=%d, "
                           "your doublet_index_offset=%d, doublet_count=%d",
                           m_total_doublet_count,
                           doublet_index_offset,
                           doublet_count);

        MUDA_KERNEL_ASSERT(subvector_offset + subvector_extent <= total_segment_count,
                           "DoubletVectorViewer: out of range, m_total_segment_count=%d, "
                           "your subvector_offset=%d, subvector_extent=%d",
                           m_total_segment_count,
                           subvector_offset,
                           subvector_extent);
    }

    // implicit conversion

    MUDA_GENERIC ConstViewer as_const() const noexcept
    {
        return ConstViewer{m_total_segment_count,
                           m_doublet_index_offset,
                           m_doublet_count,
                           m_total_doublet_count,
                           m_subvector_offset,
                           m_subvector_extent,
                           m_segment_indices,
                           m_segment_values};
    }

    MUDA_GENERIC operator ConstViewer() const noexcept { return as_const(); }

    // const access
    MUDA_GENERIC int doublet_count() const noexcept { return m_doublet_count; }
    MUDA_GENERIC int total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }

    MUDA_GENERIC CDoublet operator()(int i) const
    {
        auto index    = get_index(i);
        auto global_i = m_segment_indices[index];
        auto sub_i    = global_i - m_subvector_offset;

        check_in_subvector(sub_i);
        return CDoublet{sub_i, m_segment_values[index]};
    }

  protected:
    MUDA_INLINE MUDA_GENERIC int get_index(int i) const noexcept
    {
        MUDA_KERNEL_ASSERT(i >= 0 && i < m_doublet_count,
                           "DoubletVectorViewer [%s:%s]: index out of range, m_doublet_count=%d, your index=%d",
                           this->name(),
                           this->kernel_name(),
                           m_doublet_count,
                           i);
        auto index = i + m_doublet_index_offset;
        return index;
    }

    MUDA_INLINE MUDA_GENERIC void check_in_subvector(int i) const noexcept
    {
        MUDA_KERNEL_ASSERT(i >= 0 && i < m_subvector_extent,
                           "DoubletVectorViewer [%s:%s]: index out of range, m_subvector_extent=%d, your index=%d",
                           this->name(),
                           this->kernel_name(),
                           m_subvector_extent,
                           i);
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

    using Base::operator();

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
            return std::as_const(m_viewer).operator()(m_index);
        }

        MUDA_GENERIC void write(int segment_i, const SegmentVector& block) &&
        {
            auto index = m_viewer.get_index(m_index);

            m_viewer.check_in_subvector(segment_i);

            auto global_i = segment_i + m_viewer.m_subvector_offset;

            m_viewer.m_segment_indices[index] = global_i;
            m_viewer.m_segment_values[index]  = block;
        }

        MUDA_GENERIC ~Proxy() = default;
    };

    MUDA_GENERIC Proxy operator()(int i) { return Proxy{*this, i}; }
};

template <bool IsConst, typename T>
class DoubletVectorViewerBase<IsConst, T, 1> : public ViewerBase<IsConst>
{
    using Base = ViewerBase<IsConst>;
  protected:
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;
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
    // vector info
    int m_total_count = 0;

    // doublet info
    int m_doublet_index_offset = 0;
    int m_doublet_count        = 0;
    int m_total_doublet_count  = 0;

    // subvector info
    int m_subvector_offset = 0;
    int m_subvector_extent = 0;

    auto_const_t<int>* m_indices;
    auto_const_t<T>*   m_values;

  public:
    MUDA_GENERIC DoubletVectorViewerBase() = default;
    MUDA_GENERIC DoubletVectorViewerBase(int total_count,
                                         int doublet_index_offset,
                                         int doublet_count,
                                         int total_doublet_count,
                                         int subvector_offset,
                                         int subvector_extent,
                                         auto_const_t<int>* indices,
                                         auto_const_t<T>*   values)
        : m_total_count(total_count)
        , m_doublet_index_offset(doublet_index_offset)
        , m_doublet_count(doublet_count)
        , m_total_doublet_count(total_doublet_count)
        , m_indices(indices)
        , m_values(values)
    {
        MUDA_KERNEL_ASSERT(doublet_index_offset + doublet_count <= total_doublet_count,
                           "DoubletVectorViewer: out of range, m_total_doublet_count=%d, "
                           "your doublet_index_offset=%d, doublet_count=%d",
                           m_total_doublet_count,
                           doublet_index_offset,
                           doublet_count);

        MUDA_KERNEL_ASSERT(subvector_offset + subvector_extent <= total_count,
                           "DoubletVectorViewer: out of range, m_total_segment_count=%d, "
                           "your subvector_offset=%d, subvector_extent=%d",
                           m_total_count,
                           subvector_offset,
                           subvector_extent);
    }

    // implicit conversion

    MUDA_GENERIC ConstViewer as_const() const noexcept
    {
        return ConstViewer{m_total_count,
                           m_doublet_index_offset,
                           m_doublet_count,
                           m_total_doublet_count,
                           m_subvector_offset,
                           m_subvector_extent,
                           m_indices,
                           m_values};
    }

    MUDA_GENERIC operator ConstViewer() const noexcept { return as_const(); }

    // non-const access

    MUDA_GENERIC CDoublet operator()(int i) const
    {
        check_in_subvector(i);
        auto index    = get_index(i);
        auto global_i = m_indices[index];
        auto sub_i    = global_i - m_subvector_offset;

        return CDoublet{sub_i, m_values[index]};
    }

    MUDA_GENERIC int extent() const noexcept { return m_subvector_extent; }
    MUDA_GENERIC int total_extent() const noexcept { return m_total_count; }

    MUDA_GENERIC int subvector_offset() const noexcept
    {
        return m_subvector_offset;
    }

    MUDA_GENERIC int doublet_count() const noexcept { return m_doublet_count; }
    MUDA_GENERIC int total_doublet_count() const noexcept
    {
        return m_total_doublet_count;
    }

  protected:
    MUDA_INLINE MUDA_GENERIC int get_index(int i) const noexcept
    {

        MUDA_KERNEL_ASSERT(i >= 0 && i < m_doublet_count,
                           "DoubletVectorViewer [%s:%s]: index out of range, m_doublet_count=%d, your index=%d",
                           this->name(),
                           this->kernel_name(),
                           m_doublet_count,
                           i);
        auto index = i + m_doublet_index_offset;
        return index;
    }

    MUDA_INLINE MUDA_GENERIC void check_in_subvector(int i) const noexcept
    {
        MUDA_KERNEL_ASSERT(i >= 0 && i < m_subvector_extent,
                           "DoubletVectorViewer [%s:%s]: index out of range, m_subvector_extent=%d, your index=%d",
                           this->name(),
                           this->kernel_name(),
                           m_subvector_extent,
                           i);
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

    using Base::operator();

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
            return std::as_const(m_viewer).operator()(m_index);
        }

        MUDA_GENERIC void write(int i, const T& value) &&
        {
            m_viewer.check_in_subvector(i);

            auto index = m_viewer.get_index(m_index);

            auto global_i             = i + m_viewer.m_subvector_offset;
            m_viewer.m_indices[index] = global_i;
            m_viewer.m_values[index]  = value;
        }

        MUDA_GENERIC ~Proxy() = default;
    };

    MUDA_GENERIC Proxy operator()(int i) { return Proxy{*this, i}; }
};
}  // namespace muda

#include "details/doublet_vector_viewer.inl"
