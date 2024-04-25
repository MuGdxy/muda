#pragma once
#include <string>
#include <muda/viewer/viewer_base.h>
#include <muda/buffer/device_buffer.h>
#include <muda/tools/cuda_vec_utils.h>
#include <Eigen/Core>


/*
* - 2024/2/23 remove viewer's subview, view's subview is enough
*/

namespace muda
{
template <bool IsConst, typename T, int N>
class TripletMatrixViewerBase : public ViewerBase<IsConst>
{
    using Base = ViewerBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    using BlockMatrix    = Eigen::Matrix<T, N, N>;
    using ConstViewer    = TripletMatrixViewerBase<true, T, N>;
    using NonConstViewer = TripletMatrixViewerBase<false, T, N>;
    using ThisViewer     = TripletMatrixViewerBase<IsConst, T, N>;

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
    // matrix info
    int m_total_block_rows = 0;
    int m_total_block_cols = 0;

    // triplet info
    int m_triplet_index_offset = 0;
    int m_triplet_count        = 0;
    int m_total_triplet_count  = 0;

    // sub matrix info
    int2 m_submatrix_offset = {0, 0};
    int2 m_submatrix_extent = {0, 0};

    // data
    auto_const_t<int>*         m_block_row_indices;
    auto_const_t<int>*         m_block_col_indices;
    auto_const_t<BlockMatrix>* m_block_values;


  public:
    MUDA_GENERIC TripletMatrixViewerBase() = default;
    MUDA_GENERIC TripletMatrixViewerBase(int total_block_rows,
                                         int total_block_cols,
                                         int triplet_index_offset,
                                         int triplet_count,
                                         int total_triplet_count,

                                         int2 submatrix_offset,
                                         int2 submatrix_extent,

                                         auto_const_t<int>* block_row_indices,
                                         auto_const_t<int>* block_col_indices,
                                         auto_const_t<BlockMatrix>* block_values)
        : m_total_block_rows(total_block_rows)
        , m_total_block_cols(total_block_cols)
        , m_triplet_index_offset(triplet_index_offset)
        , m_triplet_count(triplet_count)
        , m_total_triplet_count(total_triplet_count)
        , m_submatrix_offset(submatrix_offset)
        , m_submatrix_extent(submatrix_extent)
        , m_block_row_indices(block_row_indices)
        , m_block_col_indices(block_col_indices)
        , m_block_values(block_values)
    {
        MUDA_KERNEL_ASSERT(triplet_index_offset + triplet_count <= total_triplet_count,
                           "TripletMatrixViewer [%s:%s]: out of range, m_total_triplet_count=%d, "
                           "your triplet_index_offset=%d, triplet_count=%d",
                           this->name(),
                           this->kernel_name(),
                           total_triplet_count,
                           triplet_index_offset,
                           triplet_count);

        MUDA_KERNEL_ASSERT(submatrix_offset.x >= 0 && submatrix_offset.y >= 0,
                           "TripletMatrixViewer[%s:%s]: submatrix_offset is out of range, submatrix_offset.x=%d, submatrix_offset.y=%d",
                           this->name(),
                           this->kernel_name(),
                           submatrix_offset.x,
                           submatrix_offset.y);

        MUDA_KERNEL_ASSERT(submatrix_offset.x + submatrix_extent.x <= total_block_rows,
                           "TripletMatrixViewer[%s:%s]: submatrix is out of range, submatrix_offset.x=%d, submatrix_extent.x=%d, total_block_rows=%d",
                           this->name(),
                           this->kernel_name(),
                           submatrix_offset.x,
                           submatrix_extent.x,
                           total_block_rows);

        MUDA_KERNEL_ASSERT(submatrix_offset.y + submatrix_extent.y <= total_block_cols,
                           "TripletMatrixViewer[%s:%s]: submatrix is out of range, submatrix_offset.y=%d, submatrix_extent.y=%d, total_block_cols=%d",
                           this->name(),
                           this->kernel_name(),
                           submatrix_offset.y,
                           submatrix_extent.y,
                           total_block_cols);
    }

    MUDA_GENERIC ConstViewer as_const() const
    {
        return ConstViewer{m_total_block_rows,
                           m_total_block_cols,
                           m_triplet_index_offset,
                           m_triplet_count,
                           m_total_triplet_count,
                           m_submatrix_offset,
                           m_submatrix_extent,
                           m_block_row_indices,
                           m_block_col_indices,
                           m_block_values};
    }

    MUDA_GENERIC operator ConstViewer() const { return as_const(); }

    // const accessor

    MUDA_GENERIC auto total_block_rows() const { return m_total_block_rows; }
    MUDA_GENERIC auto total_block_cols() const { return m_total_block_cols; }
    MUDA_GENERIC auto total_extent() const
    {
        return int2{m_total_block_rows, m_total_block_cols};
    }

    MUDA_GENERIC auto submatrix_offset() const { return m_submatrix_offset; }
    MUDA_GENERIC auto extent() const { return m_submatrix_extent; }

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
        auto index    = get_index(i);
        auto global_i = m_block_row_indices[index];
        auto global_j = m_block_col_indices[index];
        auto sub_i    = global_i - m_submatrix_offset.x;
        auto sub_j    = global_j - m_submatrix_offset.y;
        check_in_submatrix(sub_i, sub_j);
        return CTriplet{sub_i, sub_j, m_block_values[index]};
    }

  protected:
    MUDA_INLINE MUDA_GENERIC int get_index(int i) const noexcept
    {

        MUDA_KERNEL_ASSERT(i >= 0 && i < m_triplet_count,
                           "TripletMatrixViewer [%s:%s]: triplet_index out of range, block_count=%d, your index=%d",
                           this->name(),
                           this->kernel_name(),
                           m_triplet_count,
                           i);
        auto index = i + m_triplet_index_offset;
        return index;
    }

    MUDA_INLINE MUDA_GENERIC void check_in_submatrix(int i, int j) const noexcept
    {
        MUDA_KERNEL_ASSERT(i >= 0 && i < m_submatrix_extent.x,
                           "TripletMatrixViewer [%s:%s]: row index out of submatrix range,  submatrix_extent.x=%d, your i=%d",
                           this->name(),
                           this->kernel_name(),
                           m_submatrix_extent.x,
                           i);

        MUDA_KERNEL_ASSERT(j >= 0 && j < m_submatrix_extent.y,
                           "TripletMatrixViewer [%s:%s]: col index out of submatrix range,  submatrix_extent.y=%d, your j=%d",
                           this->name(),
                           this->kernel_name(),
                           m_submatrix_extent.y,
                           j);
    }
};

template <typename T, int N>
class CTripletMatrixViewer : public TripletMatrixViewerBase<true, T, N>
{
    using Base = TripletMatrixViewerBase<true, T, N>;
    MUDA_VIEWER_COMMON_NAME(CTripletMatrixViewer);
    using BlockMatrix = typename Base::BlockMatrix;

  public:
    using Base::Base;

    MUDA_GENERIC CTripletMatrixViewer(const Base& base)
        : Base(base)
    {
    }
};

template <typename T, int N>
class TripletMatrixViewer : public TripletMatrixViewerBase<false, T, N>
{
    using Base = TripletMatrixViewerBase<false, T, N>;
    MUDA_VIEWER_COMMON_NAME(TripletMatrixViewer);

  public:
    using Base::Base;
    using BlockMatrix    = typename Base::BlockMatrix;
    using CTriplet       = typename Base::CTriplet;
    using ConstViewer    = CTripletMatrixViewer<T, N>;
    using NonConstViewer = TripletMatrixViewer<T, N>;


    MUDA_GENERIC TripletMatrixViewer(const Base& base)
        : Base(base)
    {
    }

    using Base::operator();

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
            return std::as_const(m_viewer).operator()(m_index);
        }

        MUDA_GENERIC
        void write(int block_row_index, int block_col_index, const BlockMatrix& block) &&
        {
            auto index = m_viewer.get_index(m_index);

            m_viewer.check_in_submatrix(block_row_index, block_col_index);

            auto global_i = m_viewer.m_submatrix_offset.x + block_row_index;
            auto global_j = m_viewer.m_submatrix_offset.y + block_col_index;

            m_viewer.m_block_row_indices[index] = global_i;
            m_viewer.m_block_col_indices[index] = global_j;
            m_viewer.m_block_values[index]      = block;
        }

        MUDA_GENERIC ~Proxy() = default;
    };

    MUDA_GENERIC Proxy operator()(int i) { return Proxy{*this, i}; }
};


template <bool IsConst, typename T>
class TripletMatrixViewerBase<IsConst, T, 1> : public ViewerBase<IsConst>
{
    using Base = ViewerBase<IsConst>;
  protected:
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    using ConstViewer    = TripletMatrixViewerBase<true, T, 1>;
    using NonConstViewer = TripletMatrixViewerBase<false, T, 1>;
    using ThisViewer     = TripletMatrixViewerBase<IsConst, T, 1>;

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

  protected:
    // matrix info
    int m_total_rows = 0;
    int m_total_cols = 0;

    // triplet info
    int m_triplet_index_offset = 0;
    int m_triplet_count        = 0;
    int m_total_triplet_count  = 0;

    // sub matrix info
    int2 m_submatrix_offset = {0, 0};
    int2 m_submatrix_extent = {0, 0};

    // data
    auto_const_t<int>* m_row_indices;
    auto_const_t<int>* m_col_indices;
    auto_const_t<T>*   m_values;

  public:
    MUDA_GENERIC TripletMatrixViewerBase() = default;
    MUDA_GENERIC TripletMatrixViewerBase(int total_rows,
                                         int total_cols,

                                         int triplet_index_offset,
                                         int triplet_count,
                                         int total_triplet_count,

                                         int2 submatrix_offset,
                                         int2 submatrix_extent,

                                         auto_const_t<int>* row_indices,
                                         auto_const_t<int>* col_indices,
                                         auto_const_t<T>*   values)
        : m_total_rows(total_rows)
        , m_total_cols(total_cols)
        , m_triplet_index_offset(triplet_index_offset)
        , m_triplet_count(triplet_count)
        , m_total_triplet_count(total_triplet_count)
        , m_submatrix_offset(submatrix_offset)
        , m_submatrix_extent(submatrix_extent)
        , m_row_indices(row_indices)
        , m_col_indices(col_indices)
        , m_values(values)
    {
        MUDA_KERNEL_ASSERT(triplet_index_offset + triplet_count <= total_triplet_count,
                           "TripletMatrixViewer [%s:%s]: out of range, m_total_triplet_count=%d, "
                           "your triplet_index_offset=%d, triplet_count=%d",
                           this->name(),
                           this->kernel_name(),
                           total_triplet_count,
                           triplet_index_offset,
                           triplet_count);

        MUDA_KERNEL_ASSERT(submatrix_offset.x >= 0 && submatrix_offset.y >= 0,
                           "TripletMatrixViewer [%s:%s]: submatrix_offset is out of range, submatrix_offset.x=%d, submatrix_offset.y=%d",
                           this->name(),
                           this->kernel_name(),
                           submatrix_offset.x,
                           submatrix_offset.y);

        MUDA_KERNEL_ASSERT(submatrix_offset.x + submatrix_extent.x <= total_rows,
                           "TripletMatrixViewer [%s:%s]: submatrix is out of range, submatrix_offset.x=%d, submatrix_extent.x=%d, rows=%d",
                           this->name(),
                           this->kernel_name(),
                           submatrix_offset.x,
                           submatrix_extent.x,
                           total_rows);

        MUDA_KERNEL_ASSERT(submatrix_offset.y + submatrix_extent.y <= total_cols,
                           "TripletMatrixViewer [%s:%s]: submatrix is out of range, submatrix_offset.y=%d, submatrix_extent.y=%d, cols=%d",
                           this->name(),
                           this->kernel_name(),
                           submatrix_offset.y,
                           submatrix_extent.y,
                           total_cols);
    }

    // implicit conversion

    MUDA_GENERIC ConstViewer as_const() const
    {
        return ConstViewer{m_total_rows,
                           m_total_cols,
                           m_triplet_index_offset,
                           m_triplet_count,
                           m_total_triplet_count,
                           m_submatrix_offset,
                           m_submatrix_extent,
                           m_row_indices,
                           m_col_indices,
                           m_values};
    }

    MUDA_GENERIC operator ConstViewer() const { return as_const(); }


    MUDA_GENERIC CTriplet operator()(int i) const
    {
        auto index = get_index(i);

        auto global_i = m_row_indices[index];
        auto global_j = m_col_indices[index];
        auto sub_i    = global_i - m_submatrix_offset.x;
        auto sub_j    = global_j - m_submatrix_offset.y;
        check_in_submatrix(sub_i, sub_j);
        return CTriplet{sub_i, sub_j, m_values[index]};
    }

    auto total_rows() const { return m_total_rows; }
    auto total_cols() const { return m_total_cols; }

    auto triplet_count() const { return m_triplet_count; }
    auto tripet_index_offset() const { return m_triplet_index_offset; }
    auto total_triplet_count() const { return m_total_triplet_count; }

    auto submatrix_offset() const { return m_submatrix_offset; }
    auto extent() const { return m_submatrix_extent; }
    auto total_extent() const { return int2{m_total_rows, m_total_cols}; }

  protected:
    MUDA_INLINE MUDA_GENERIC int get_index(int i) const noexcept
    {

        MUDA_KERNEL_ASSERT(i >= 0 && i < m_triplet_count,
                           "TripletMatrixViewer [%s:%s]: triplet_index out of range, block_count=%d, your index=%d",
                           this->name(),
                           this->kernel_name(),
                           m_triplet_count,
                           i);
        auto index = i + m_triplet_index_offset;
        return index;
    }

    MUDA_INLINE MUDA_GENERIC void check_in_submatrix(int i, int j) const noexcept
    {
        MUDA_KERNEL_ASSERT(i >= 0 && i < m_submatrix_extent.x,
                           "TripletMatrixViewer [%s:%s]: row index out of submatrix range, submatrix_extent.x=%d, yours=%d",
                           this->name(),
                           this->kernel_name(),
                           m_submatrix_extent.x,
                           i);

        MUDA_KERNEL_ASSERT(j >= 0 && j < m_submatrix_extent.y,
                           "TripletMatrixViewer [%s:%s]: col index out of submatrix range, submatrix_extent.y=%d, yours=%d",
                           this->name(),
                           this->kernel_name(),
                           m_submatrix_extent.y,
                           j);
    }
};

template <typename T>
class CTripletMatrixViewer<T, 1> : public TripletMatrixViewerBase<true, T, 1>
{
    using Base = TripletMatrixViewerBase<true, T, 1>;
    MUDA_VIEWER_COMMON_NAME(CTripletMatrixViewer);

  public:
    using Base::Base;
    using ConstViewer = CTripletMatrixViewer<T, 1>;

    MUDA_GENERIC CTripletMatrixViewer(const Base& base)
        : Base(base)
    {
    }
};

template <typename T>
class TripletMatrixViewer<T, 1> : public TripletMatrixViewerBase<false, T, 1>
{
    using Base = TripletMatrixViewerBase<false, T, 1>;
    MUDA_VIEWER_COMMON_NAME(TripletMatrixViewer);

  public:
    using ConstViewer    = CTripletMatrixViewer<T, 1>;
    using NonConstViewer = TripletMatrixViewer<T, 1>;

    using Base::Base;
    using CTriplet = typename Base::CTriplet;
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
            return std::as_const(m_viewer).operator()(m_index);
        }

        MUDA_GENERIC void write(int row_index, int col_index, const T& value) &&
        {
            auto index = m_viewer.get_index(m_index);
            m_viewer.check_in_submatrix(row_index, col_index);

            auto global_i = m_viewer.m_submatrix_offset.x + row_index;
            auto global_j = m_viewer.m_submatrix_offset.y + col_index;

            m_viewer.m_row_indices[index] = global_i;
            m_viewer.m_col_indices[index] = global_j;
            m_viewer.m_values[index]      = value;
        }

        MUDA_GENERIC ~Proxy() = default;
    };

    using Base::operator();

    MUDA_GENERIC Proxy operator()(int i)
    {
        auto index = Base::get_index(i);
        return Proxy{*this, index};
    }
};
}  // namespace muda

#include "details/triplet_matrix_viewer.inl"
