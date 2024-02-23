#pragma once
#include <muda/buffer/buffer_view.h>
#include <muda/ext/linear_system/triplet_matrix_viewer.h>
#include <muda/view/view_base.h>

namespace muda
{
template <bool IsConst, typename Ty, int N>
class TripletMatrixViewBase : public ViewBase<IsConst>
{
  public:
    static_assert(!std::is_const_v<Ty>, "Ty must be non-const");
    using ConstView    = TripletMatrixViewBase<true, Ty, N>;
    using NonConstView = TripletMatrixViewBase<false, Ty, N>;
    using ThisView     = TripletMatrixViewBase<IsConst, Ty, N>;

  private:
    using CViewer    = CTripletMatrixViewer<Ty, N>;
    using Viewer     = TripletMatrixViewer<Ty, N>;
    using ThisViewer = std::conditional_t<IsConst, CViewer, Viewer>;

  public:
    using BlockMatrix = Eigen::Matrix<Ty, N, N>;

  protected:
    // matrix info
    int m_total_block_rows = 0;
    int m_total_block_cols = 0;

    // writable range: TODO
    int2 m_sub_begin = {0, 0};
    int2 m_sub_end   = {0, 0};

    // triplet info
    int m_triplet_index_offset = 0;
    int m_triplet_count        = 0;
    int m_total_triplet_count  = 0;

    auto_const_t<int>*         m_block_row_indices = nullptr;
    auto_const_t<int>*         m_block_col_indices = nullptr;
    auto_const_t<BlockMatrix>* m_block_values      = nullptr;

  public:
    MUDA_GENERIC TripletMatrixViewBase() = default;
    MUDA_GENERIC TripletMatrixViewBase(int                rows,
                                       int                cols,
                                       int                triplet_index_offset,
                                       int                triplet_count,
                                       int                total_triplet_count,
                                       auto_const_t<int>* block_row_indices,
                                       auto_const_t<int>* block_col_indices,
                                       auto_const_t<BlockMatrix>* block_values)
        : m_total_block_rows(rows)
        , m_total_block_cols(cols)
        , m_triplet_index_offset(triplet_index_offset)
        , m_triplet_count(triplet_count)
        , m_total_triplet_count(total_triplet_count)
        , m_block_row_indices(block_row_indices)
        , m_block_col_indices(block_col_indices)
        , m_block_values(block_values)
        , m_sub_begin(0, 0)
        , m_sub_end(rows, cols)
    {
        MUDA_ASSERT(triplet_index_offset + triplet_count <= total_triplet_count,
                    "TripletMatrixView: out of range, m_total_triplet_count=%d, "
                    "your triplet_index_offset=%d, triplet_count=%d",
                    total_triplet_count,
                    triplet_index_offset,
                    triplet_count);
    }

    // explicit conversion to non-const
    MUDA_GENERIC ConstView as_const() const
    {
        return ConstView{m_total_block_rows,
                         m_total_block_cols,
                         m_triplet_index_offset,
                         m_triplet_count,
                         m_total_triplet_count,
                         m_block_row_indices,
                         m_block_col_indices,
                         m_block_values};
    }

    // implicit conversion to const
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_GENERIC auto subview(int offset, int count) const
    {
        return ThisView{m_total_block_rows,
                        m_total_block_cols,
                        m_triplet_index_offset + offset,
                        count,
                        m_total_triplet_count,
                        m_block_row_indices,
                        m_block_col_indices,
                        m_block_values};
    }

    MUDA_GENERIC auto subview(int offset) const
    {
        MUDA_ASSERT(offset < m_triplet_count,
                    "TripletMatrixView: offset is out of range, size=%d, your offset=%d",
                    m_triplet_count,
                    offset);
        return subview(offset, m_triplet_count - offset);
    }

    MUDA_GENERIC auto cviewer() const
    {
        return CViewer{m_total_block_rows,
                       m_total_block_cols,
                       m_triplet_index_offset,
                       m_triplet_count,
                       m_total_triplet_count,
                       m_block_row_indices,
                       m_block_col_indices,
                       m_block_values};
    }

    MUDA_GENERIC auto viewer()
    {
        return ThisViewer{m_total_block_rows,
                          m_total_block_cols,
                          m_triplet_index_offset,
                          m_triplet_count,
                          m_total_triplet_count,
                          m_block_row_indices,
                          m_block_col_indices,
                          m_block_values};
    }

    // non-const access
    auto_const_t<BlockMatrix>* block_values() { return m_block_values; }
    auto_const_t<int>* block_row_indices() { return m_block_row_indices; }
    auto_const_t<int>* block_col_indices() { return m_block_col_indices; }

    auto sub_block(int2 offset, int2 size)
    {
        MUDA_ASSERT(m_sub_begin.x + offset.x + size.x <= m_sub_end,
                    "TripletMatrixView: sub block is out of range, sub_begin.x=%d, offset.x=%d, size.x=%d, sub_end.x=%d",
                    m_sub_begin.x,
                    offset.x,
                    size.x,
                    m_sub_end.x);

        MUDA_ASSERT(m_sub_begin.y + offset.y + size.y <= m_sub_end,
                    "TripletMatrixView: sub block is out of range, sub_begin.y=%d, offset.y=%d, size.y=%d, sub_end.y=%d",
                    m_sub_begin.y,
                    offset.y,
                    size.y,
                    m_sub_end.y);

        auto copy = *this;
        copy.m_sub_begin.x += offset.x;
        copy.m_sub_begin.y += offset.y;
        copy.m_sub_end.x = copy.m_sub_begin.x + size.x;
        copy.m_sub_end.y = copy.m_sub_begin.y + size.y;

        return copy;
    }

    auto sub_block(int2 offset, int2 size) const
    {
        return as_const().sub_block(offset, size);
    }

    // const access
    auto block_values() const { return m_block_values; }
    auto block_row_indices() const { return m_block_row_indices; }
    auto block_col_indices() const { return m_block_col_indices; }

    auto total_block_rows() const { return m_total_block_rows; }
    auto total_block_cols() const { return m_total_block_cols; }

    auto triplet_count() const { return m_triplet_count; }
    auto tripet_index_offset() const { return m_triplet_index_offset; }
    auto total_triplet_count() const { return m_total_triplet_count; }
};

template <bool IsConst, typename Ty>
class TripletMatrixViewBase<IsConst, Ty, 1> : public ViewBase<IsConst>
{
  public:
    static_assert(!std::is_const_v<Ty>, "Ty must be non-const");
    using ConstView    = TripletMatrixViewBase<true, Ty, 1>;
    using NonConstView = TripletMatrixViewBase<false, Ty, 1>;
    using ThisView     = TripletMatrixViewBase<IsConst, Ty, 1>;

  private:
    using CViewer    = CTripletMatrixViewer<Ty, 1>;
    using Viewer     = TripletMatrixViewer<Ty, 1>;
    using ThisViewer = std::conditional_t<IsConst, CViewer, Viewer>;

  protected:
    // matrix info
    int m_rows = 0;
    int m_cols = 0;

    int2 m_sub_begin = {0, 0};
    int2 m_sub_end   = {0, 0};

    // triplet info
    int m_triplet_index_offset = 0;
    int m_triplet_count        = 0;
    int m_total_triplet_count  = 0;

    // data
    auto_const_t<int>* m_row_indices;
    auto_const_t<int>* m_col_indices;
    auto_const_t<Ty>*  m_values;


  public:
    MUDA_GENERIC TripletMatrixViewBase() = default;

    MUDA_GENERIC TripletMatrixViewBase(int                rows,
                                       int                cols,
                                       int                triplet_index_offset,
                                       int                triplet_count,
                                       int                total_triplet_count,
                                       auto_const_t<int>* row_indices,
                                       auto_const_t<int>* col_indices,
                                       auto_const_t<Ty>*  values)
        : m_rows(rows)
        , m_cols(cols)
        , m_triplet_index_offset(triplet_index_offset)
        , m_triplet_count(triplet_count)
        , m_total_triplet_count(total_triplet_count)
        , m_row_indices(row_indices)
        , m_col_indices(col_indices)
        , m_values(values)
        , m_sub_begin(0, 0)
        , m_sub_end(rows, cols)
    {
        MUDA_ASSERT(triplet_index_offset + triplet_count <= total_triplet_count,
                    "TripletMatrixView: out of range, m_total_triplet_count=%d, "
                    "your triplet_index_offset=%d, triplet_count=%d",
                    total_triplet_count,
                    triplet_index_offset,
                    triplet_count);
    }

    // explicit conversion to non-const
    MUDA_GENERIC ConstView as_const() const
    {
        return ConstView{m_rows,
                         m_cols,
                         m_triplet_index_offset,
                         m_triplet_count,
                         m_total_triplet_count,
                         m_row_indices,
                         m_col_indices,
                         m_values};
    }

    // implicit conversion to const
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_GENERIC auto subview(int offset, int count) const
    {
        return ThisView{m_rows,
                        m_cols,
                        m_triplet_index_offset + offset,
                        count,
                        m_total_triplet_count,
                        m_row_indices,
                        m_col_indices,
                        m_values};
    }

    MUDA_GENERIC auto subview(int offset) const
    {
        MUDA_ASSERT(offset < m_triplet_count,
                    "TripletMatrixView: offset is out of range, size=%d, your offset=%d",
                    m_triplet_count,
                    offset);
        return subview(offset, m_triplet_count - offset);
    }

    MUDA_GENERIC auto cviewer() const
    {
        return CViewer{m_rows,
                       m_cols,
                       m_triplet_index_offset,
                       m_triplet_count,
                       m_total_triplet_count,
                       m_row_indices,
                       m_col_indices,
                       m_values};
    }

    MUDA_GENERIC auto viewer()
    {
        return ThisViewer{m_rows,
                          m_cols,
                          m_triplet_index_offset,
                          m_triplet_count,
                          m_total_triplet_count,
                          m_row_indices,
                          m_col_indices,
                          m_values};
    }

    // non-const access
    auto_const_t<Ty>*  values() { return m_values; }
    auto_const_t<int>* row_indices() { return m_row_indices; }
    auto_const_t<int>* col_indices() { return m_col_indices; }


    // const access
    auto values() const { return m_values; }
    auto row_indices() const { return m_row_indices; }
    auto col_indices() const { return m_col_indices; }

    auto rows() const { return m_rows; }
    auto cols() const { return m_cols; }
    auto triplet_count() const { return m_triplet_count; }
    auto tripet_index_offset() const { return m_triplet_index_offset; }
    auto total_triplet_count() const { return m_total_triplet_count; }
};

template <typename Ty, int N>
using TripletMatrixView = TripletMatrixViewBase<false, Ty, N>;
template <typename Ty, int N>
using CTripletMatrixView = TripletMatrixViewBase<true, Ty, N>;
}  // namespace muda

namespace muda
{
template <typename Ty, int N>
struct read_only_viewer<TripletMatrixView<Ty, N>>
{
    using type = CTripletMatrixView<Ty, N>;
};

template <typename Ty, int N>
struct read_write_viewer<TripletMatrixView<Ty, N>>
{
    using type = TripletMatrixView<Ty, N>;
};
}  // namespace muda


#include "details/triplet_matrix_view.inl"
