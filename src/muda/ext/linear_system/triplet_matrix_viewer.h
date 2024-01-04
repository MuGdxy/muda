#pragma once
#include <string>
#include <muda/viewer/viewer_base.h>
#include <muda/buffer/device_buffer.h>
#include <Eigen/Core>

namespace muda
{
template <typename T, int N>
class TripletMatrixViewerBase : public muda::ViewerBase
{
  public:
    using BlockMatrix = Eigen::Matrix<T, N, N>;

    struct CTriplet
    {
        MUDA_GENERIC CTriplet(int row_index, int col_index, const BlockMatrix& block)
            : block_row_index(row_index)
            , block_col_index(col_index)
            , block_value(block)
        {
        }
        int                block_row_index;
        int                block_col_index;
        const BlockMatrix& block_value;
    };


  protected:
    // data
    int*         m_block_row_indices;
    int*         m_block_col_indices;
    BlockMatrix* m_block_values;

    // matrix info
    int m_block_rows = 0;
    int m_block_cols = 0;

    // triplet info
    int m_triplet_index_offset = 0;
    int m_triplet_count        = 0;
    int m_total_triplet_count  = 0;

  public:
    MUDA_GENERIC TripletMatrixViewerBase() = default;
    MUDA_GENERIC TripletMatrixViewerBase(int          rows,
                                         int          cols,
                                         int          triplet_index_offset,
                                         int          triplet_count,
                                         int          total_triplet_count,
                                         int*         block_row_indices,
                                         int*         block_col_indices,
                                         BlockMatrix* block_values)
        : m_block_rows(rows)
        , m_block_cols(cols)
        , m_triplet_index_offset(triplet_index_offset)
        , m_triplet_count(triplet_count)
        , m_total_triplet_count(total_triplet_count)
        , m_block_row_indices(block_row_indices)
        , m_block_col_indices(block_col_indices)
        , m_block_values(block_values)
    {
        MUDA_KERNEL_ASSERT(triplet_index_offset + triplet_count <= total_triplet_count,
                           "TripletMatrixViewer [%s:%s]: out of range, m_total_triplet_count=%d, "
                           "your triplet_index_offset=%d, triplet_count=%d",
                           name(),
                           kernel_name(),
                           total_triplet_count,
                           triplet_index_offset,
                           triplet_count);
    }

    MUDA_GENERIC auto block_rows() const { return m_block_rows; }
    MUDA_GENERIC auto block_cols() const { return m_block_cols; }
    MUDA_GENERIC auto triplet_count() const { return m_triplet_count; }
    MUDA_GENERIC auto tripet_index_offset() const
    {
        return m_triplet_index_offset;
    }
    MUDA_GENERIC auto total_triplet_count() const
    {
        return m_total_triplet_count;
    }

    MUDA_GENERIC CTriplet operator()(int i) const
    {
        auto index = get_index(i);
        return CTriplet{m_block_row_indices[index],
                        m_block_col_indices[index],
                        m_block_values[index]};
    }

    MUDA_GENERIC TripletMatrixViewerBase<T, N> sub_view(int offset, int count) const
    {
        return TripletMatrixViewerBase{m_block_rows,
                                       m_block_cols,
                                       m_triplet_index_offset + offset,
                                       count,
                                       m_total_triplet_count,
                                       m_block_row_indices,
                                       m_block_col_indices,
                                       m_block_values};
    }

    MUDA_GENERIC TripletMatrixViewerBase<T, N> sub_view(int offset) const
    {
        MUDA_KERNEL_ASSERT(offset < m_triplet_count,
                           "TripletMatrixViewer [%s:%s]: offset is out of range, size=%d, your offset=%d",
                           name(),
                           kernel_name(),
                           m_triplet_count,
                           offset);
        return sub_view(offset, m_triplet_count - offset);
    }

  protected:
    MUDA_INLINE MUDA_GENERIC int get_index(int i) const noexcept
    {

        MUDA_KERNEL_ASSERT(i >= 0 && i < m_triplet_count,
                           "TripletMatrixViewer [%s:%s]: triplet_index out of range, block_count=%d, your index=%d",
                           name(),
                           kernel_name(),
                           m_triplet_count,
                           i);
        auto index = i + m_triplet_index_offset;
        return index;
    }
};

template <typename T, int N>
class CTripletMatrixViewer : public TripletMatrixViewerBase<T, N>
{
    using Base = TripletMatrixViewerBase<T, N>;
    MUDA_VIEWER_COMMON_NAME(CTripletMatrixViewer);

  public:
    MUDA_GENERIC CTripletMatrixViewer(int                rows,
                                      int                cols,
                                      int                triplet_index_offset,
                                      int                triplet_count,
                                      int                total_triplet_count,
                                      const int*         block_row_indices,
                                      const int*         block_col_indices,
                                      const BlockMatrix* block_values)
        : Base(rows,
               cols,
               triplet_index_offset,
               triplet_count,
               total_triplet_count,
               const_cast<int*>(block_row_indices),
               const_cast<int*>(block_col_indices),
               const_cast<BlockMatrix*>(block_values))
    {
    }

    MUDA_GENERIC CTripletMatrixViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CTripletMatrixViewer<T, N> sub_view(int offset, int count) const
    {
        return CTripletMatrixViewer{Base::sub_view(offset, count)};
    }

    MUDA_GENERIC CTripletMatrixViewer<T, N> sub_view(int offset) const
    {
        return CTripletMatrixViewer{Base::sub_view(offset)};
    }
};

template <typename T, int N>
class TripletMatrixViewer : public TripletMatrixViewerBase<T, N>
{
    using Base = TripletMatrixViewerBase<T, N>;
    MUDA_VIEWER_COMMON_NAME(TripletMatrixViewer);

  public:
    using Base::Base;
    MUDA_GENERIC TripletMatrixViewer(const Base& base)
        : Base(base)
    {
    }

    class Proxy
    {
        friend class TripletMatrixViewer;
        TripletMatrixViewer& m_viewer;
        int                  m_index = 0;

      private:
        MUDA_GENERIC Proxy(TripletMatrixViewer& viewer, int index)
            : m_viewer(viewer)
            , m_index(index)
        {
        }

      public:
        MUDA_GENERIC auto read() &&
        {
            return CTriplet{m_viewer.m_block_row_indices[m_index],
                            m_viewer.m_block_col_indices[m_index],
                            m_viewer.m_block_values[m_index]};
        }

        MUDA_GENERIC void write(int block_row_index, int block_col_index, const BlockMatrix& block) &&
        {
            MUDA_KERNEL_ASSERT(block_row_index >= 0
                                   && block_row_index < m_viewer.m_block_rows,
                               "TripletMatrixViewer [%s:%s]: block_row_index out of range, m_block_rows=%d, yours=%d",
                               m_viewer.name(),
                               m_viewer.kernel_name(),
                               m_viewer.m_block_rows,
                               block_row_index);

            MUDA_KERNEL_ASSERT(block_col_index >= 0
                                   && block_col_index < m_viewer.m_block_cols,
                               "TripletMatrixViewer [%s:%s]: block_col_index out of range, m_block_cols=%d, yours=%d",
                               m_viewer.name(),
                               m_viewer.kernel_name(),
                               m_viewer.m_block_cols,
                               block_col_index);

            m_viewer.m_block_row_indices[m_index] = block_row_index;
            m_viewer.m_block_col_indices[m_index] = block_col_index;
            m_viewer.m_block_values[m_index]      = block;
        }

        MUDA_GENERIC ~Proxy() = default;
    };

    using Base::operator();

    MUDA_GENERIC Proxy operator()(int i)
    {
        auto index = Base::get_index(i);
        return Proxy{*this, index};
    }

    MUDA_GENERIC TripletMatrixViewer<T, N> sub_view(int offset, int count) const
    {
        return TripletMatrixViewer{Base::sub_view(offset, count)};
    }

    MUDA_GENERIC TripletMatrixViewer<T, N> sub_view(int offset) const
    {
        return TripletMatrixViewer{Base::sub_view(offset)};
    }
};


template <typename T>
class TripletMatrixViewerBase<T, 1> : public muda::ViewerBase
{
  public:
    struct CTriplet
    {
        MUDA_GENERIC CTriplet(int row_index, int col_index, const T& block)
            : row_index(row_index)
            , col_index(col_index)
            , value(block)
        {
        }
        int      row_index;
        int      col_index;
        const T& value;
    };

  private:
    // data
    int* m_row_indices;
    int* m_col_indices;
    T*   m_values;

    // matrix info
    int m_rows = 0;
    int m_cols = 0;

    // triplet info
    int m_triplet_index_offset = 0;
    int m_triplet_count        = 0;
    int m_total_triplet_count  = 0;

  public:
    MUDA_GENERIC TripletMatrixViewerBase() = default;
    MUDA_GENERIC TripletMatrixViewerBase(int  rows,
                                         int  cols,
                                         int  triplet_index_offset,
                                         int  triplet_count,
                                         int  total_triplet_count,
                                         int* row_indices,
                                         int* col_indices,
                                         T*   values)
        : m_rows(rows)
        , m_cols(cols)
        , m_triplet_index_offset(triplet_index_offset)
        , m_triplet_count(triplet_count)
        , m_total_triplet_count(total_triplet_count)
        , m_row_indices(row_indices)
        , m_col_indices(col_indices)
        , m_values(values)
    {
        MUDA_KERNEL_ASSERT(triplet_index_offset + triplet_count <= total_triplet_count,
                           "TripletMatrixViewer [%s:%s]: out of range, m_total_triplet_count=%d, "
                           "your triplet_index_offset=%d, triplet_count=%d",
                           name(),
                           kernel_name(),
                           total_triplet_count,
                           triplet_index_offset,
                           triplet_count);
    }

    MUDA_GENERIC auto block_rows() const { return m_rows; }
    MUDA_GENERIC auto block_cols() const { return m_cols; }
    MUDA_GENERIC auto triplet_count() const { return m_triplet_count; }
    MUDA_GENERIC auto tripet_index_offset() const
    {
        return m_triplet_index_offset;
    }
    MUDA_GENERIC auto total_triplet_count() const
    {
        return m_total_triplet_count;
    }


    MUDA_GENERIC void write(int i, int block_row_index, int block_col_index, const T& block)
    {
        MUDA_ASSERT(block_row_index >= 0 && block_row_index < m_rows,
                    "TripletMatrixViewer [%s:%s]: block_row_index out of range, m_block_rows=%d, yours=%d",
                    name(),
                    kernel_name(),
                    m_rows,
                    block_row_index);
        MUDA_ASSERT(block_col_index >= 0 && block_col_index < m_cols,
                    "TripletMatrixViewer [%s:%s]: block_col_index out of range, m_block_cols=%d, yours=%d",
                    name(),
                    kernel_name(),
                    m_cols,
                    block_col_index);

        auto index = get_index(i);

        m_row_indices[index] = block_row_index;
        m_col_indices[index] = block_col_index;
        m_values[index]      = block;
    }

    MUDA_GENERIC CTriplet operator()(int i) const
    {
        auto index = get_index(i);
        return CTriplet{m_row_indices[index], m_col_indices[index], m_values[index]};
    }

    MUDA_GENERIC TripletMatrixViewerBase<T, 1> sub_view(int offset, int count) const
    {
        return TripletMatrixViewerBase{m_rows,
                                       m_cols,
                                       m_triplet_index_offset + offset,
                                       count,
                                       m_total_triplet_count,
                                       m_row_indices,
                                       m_col_indices,
                                       m_values};
    }

    MUDA_GENERIC TripletMatrixViewerBase<T, 1> sub_view(int offset) const
    {
        MUDA_KERNEL_ASSERT(offset < m_triplet_count,
                           "TripletMatrixViewer [%s:%s]: offset is out of range, size=%d, your offset=%d",
                           name(),
                           kernel_name(),
                           m_triplet_count,
                           offset);
        return sub_view(offset, m_triplet_count - offset);
    }

  private:
    MUDA_INLINE MUDA_GENERIC int get_index(int i) const noexcept
    {

        MUDA_KERNEL_ASSERT(i >= 0 && i < m_triplet_count,
                           "TripletMatrixViewer [%s:%s]: triplet_index out of range, block_count=%d, your index=%d",
                           name(),
                           kernel_name(),
                           m_triplet_count,
                           i);
        auto index = i + m_triplet_index_offset;
        return index;
    }
};

template <typename T>
class CTripletMatrixViewer<T, 1> : public TripletMatrixViewerBase<T, 1>
{
    using Base = TripletMatrixViewerBase<T, 1>;
    MUDA_VIEWER_COMMON_NAME(CTripletMatrixViewer);

  public:
    MUDA_GENERIC CTripletMatrixViewer(int        rows,
                                      int        cols,
                                      int        triplet_index_offset,
                                      int        triplet_count,
                                      int        total_triplet_count,
                                      const int* row_indices,
                                      const int* col_indices,
                                      const T*   values)
        : Base(rows,
               cols,
               triplet_index_offset,
               triplet_count,
               total_triplet_count,
               const_cast<int*>(row_indices),
               const_cast<int*>(col_indices),
               const_cast<T*>(values))
    {
    }

    MUDA_GENERIC CTripletMatrixViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CTripletMatrixViewer<T, 1> sub_view(int offset, int count) const
    {
        return CTripletMatrixViewer{Base::sub_view(offset, count)};
    }

    MUDA_GENERIC CTripletMatrixViewer<T, 1> sub_view(int offset) const
    {
        return CTripletMatrixViewer{Base::sub_view(offset)};
    }
};

template <typename T>
class TripletMatrixViewer<T, 1> : public TripletMatrixViewerBase<T, 1>
{
    using Base = TripletMatrixViewerBase<T, 1>;
    MUDA_VIEWER_COMMON_NAME(TripletMatrixViewer);

  public:
    using Base::Base;
    MUDA_GENERIC TripletMatrixViewer(const Base& base)
        : Base(base)
    {
    }

    class Proxy
    {
        friend class TripletMatrixViewer;
        TripletMatrixViewer& m_viewer;
        int                  m_index = 0;

      private:
        MUDA_GENERIC Proxy(TripletMatrixViewer& viewer, int index)
            : m_viewer(viewer)
            , m_index(index)
        {
        }

      public:
        MUDA_GENERIC auto read() &&
        {
            return CTriplet{m_viewer.m_row_indices[m_index],
                            m_viewer.m_col_indices[m_index],
                            m_viewer.m_values[m_index]};
        }

        MUDA_GENERIC void write(int row_index, int col_index, const T& value) &&
        {
            MUDA_KERNEL_ASSERT(row_index >= 0 && row_index < m_viewer.m_rows,
                               "TripletMatrixViewer [%s:%s]: row_index out of range, m_rows=%d, yours=%d",
                               m_viewer.name(),
                               m_viewer.kernel_name(),
                               m_viewer.m_rows,
                               row_index);

            MUDA_KERNEL_ASSERT(col_index >= 0 && col_index < m_viewer.m_cols,
                               "TripletMatrixViewer [%s:%s]: col_index out of range, m_cols=%d, yours=%d",
                               m_viewer.name(),
                               m_viewer.kernel_name(),
                               m_viewer.m_cols,
                               col_index);

            m_viewer.m_row_indices[m_index] = row_index;
            m_viewer.m_col_indices[m_index] = col_index;
            m_viewer.m_values[m_index]      = value;
        }

        MUDA_GENERIC ~Proxy() = default;
    };

    using Base::operator();

    MUDA_GENERIC Proxy operator()(int i)
    {
        auto index = Base::get_index(i);
        return Proxy{*this, index};
    }

    MUDA_GENERIC TripletMatrixViewer<T, 1> sub_view(int offset, int count) const
    {
        return TripletMatrixViewer{Base::sub_view(offset, count)};
    }

    MUDA_GENERIC TripletMatrixViewer<T, 1> sub_view(int offset) const
    {
        return TripletMatrixViewer{Base::sub_view(offset)};
    }
};

}  // namespace muda

#include "details/triplet_matrix_viewer.inl"
