#pragma once
#include <muda/buffer/buffer_view.h>
#include <muda/ext/linear_system/triplet_matrix_viewer.h>

//namespace muda
//{
//template <typename T, int N>
//class TripletMatrixViewBase
//{
//  public:
//    using BlockMatrix = Eigen::Matrix<T, N, N>;
//
//  protected:
//    // data
//    int*         m_block_row_indices = nullptr;
//    int*         m_block_col_indices = nullptr;
//    BlockMatrix* m_block_values      = nullptr;
//
//    // matrix info
//    int m_block_rows = 0;
//    int m_block_cols = 0;
//
//    // triplet info
//    int m_triplet_index_offset = 0;
//    int m_triplet_count        = 0;
//    int m_total_triplet_count  = 0;
//
//  public:
//    MUDA_GENERIC TripletMatrixViewBase() = default;
//    MUDA_GENERIC TripletMatrixViewBase(int          rows,
//                                       int          cols,
//                                       int          triplet_index_offset,
//                                       int          triplet_count,
//                                       int          total_triplet_count,
//                                       int*         block_row_indices,
//                                       int*         block_col_indices,
//                                       BlockMatrix* block_values)
//        : m_block_rows(rows)
//        , m_block_cols(cols)
//        , m_triplet_index_offset(triplet_index_offset)
//        , m_triplet_count(triplet_count)
//        , m_total_triplet_count(total_triplet_count)
//        , m_block_row_indices(block_row_indices)
//        , m_block_col_indices(block_col_indices)
//        , m_block_values(block_values)
//    {
//        MUDA_ASSERT(triplet_index_offset + triplet_count <= total_triplet_count,
//                    "TripletMatrixView: out of range, m_total_triplet_count=%d, "
//                    "your triplet_index_offset=%d, triplet_count=%d",
//                    total_triplet_count,
//                    triplet_index_offset,
//                    triplet_count);
//    }
//
//    MUDA_GENERIC TripletMatrixViewBase<T, N> subview(int offset, int count) const
//    {
//        return TripletMatrixViewBase<T, N>{m_block_rows,
//                                           m_block_cols,
//                                           m_triplet_index_offset + offset,
//                                           count,
//                                           m_total_triplet_count,
//                                           m_block_row_indices,
//                                           m_block_col_indices,
//                                           m_block_values};
//    }
//
//    MUDA_GENERIC TripletMatrixViewBase<T, N> subview(int offset) const
//    {
//        MUDA_ASSERT(offset < m_triplet_count,
//                    "TripletMatrixView: offset is out of range, size=%d, your offset=%d",
//                    m_triplet_count,
//                    offset);
//        return subview(offset, m_triplet_count - offset);
//    }
//
//    auto cviewer() const
//    {
//        return CTripletMatrixViewer<T, N>{m_block_rows,
//                                          m_block_cols,
//                                          m_triplet_index_offset,
//                                          m_triplet_count,
//                                          m_total_triplet_count,
//                                          m_block_row_indices,
//                                          m_block_col_indices,
//                                          m_block_values};
//    }
//};
//
//template <typename T, int N>
//class CTripletMatrixView : public TripletMatrixViewBase<T, N>
//{
//    using Base = TripletMatrixViewBase<T, N>;
//
//  public:
//    using BlockMatrix = typename Base::BlockMatrix;
//    MUDA_GENERIC CTripletMatrixView(int                rows,
//                                    int                cols,
//                                    int                triplet_index_offset,
//                                    int                triplet_count,
//                                    int                total_triplet_count,
//                                    const int*         block_row_indices,
//                                    const int*         block_col_indices,
//                                    const BlockMatrix* block_values)
//        : Base(rows,
//               cols,
//               triplet_index_offset,
//               triplet_count,
//               total_triplet_count,
//               const_cast<int*>(block_row_indices),
//               const_cast<int*>(block_col_indices),
//               const_cast<BlockMatrix*>(block_values))
//    {
//    }
//
//    MUDA_GENERIC CTripletMatrixView(const Base& base)
//        : Base(base)
//    {
//    }
//
//    MUDA_GENERIC CTripletMatrixView<T, N> subview(int offset, int count) const
//    {
//        return CTripletMatrixView{Base::subview(offset, count)};
//    }
//
//    MUDA_GENERIC CTripletMatrixView<T, N> subview(int offset) const
//    {
//        return CTripletMatrixView{Base::subview(offset)};
//    }
//
//    using Base::cviewer;
//};
//
//template <typename T, int N>
//class TripletMatrixView : public TripletMatrixViewBase<T, N>
//{
//    using Base = TripletMatrixViewBase<T, N>;
//
//  public:
//    using Base::Base;
//    MUDA_GENERIC TripletMatrixView(const Base& base)
//        : Base(base)
//    {
//    }
//
//    MUDA_GENERIC TripletMatrixView(const CTripletMatrixView<T, N>&) = delete;
//
//    MUDA_GENERIC TripletMatrixView<T, N> subview(int offset, int count) const
//    {
//        return TripletMatrixView{Base::subview(offset, count)};
//    }
//
//    MUDA_GENERIC TripletMatrixView<T, N> subview(int offset) const
//    {
//        return TripletMatrixView{Base::subview(offset)};
//    }
//
//    using Base::cviewer;
//    auto viewer() const
//    {
//        return TripletMatrixViewer<T, N>{m_block_rows,
//                                         m_block_cols,
//                                         m_triplet_index_offset,
//                                         m_triplet_count,
//                                         m_total_triplet_count,
//                                         m_block_row_indices,
//                                         m_block_col_indices,
//                                         m_block_values};
//    }
//};
//
//template <typename T>
//class TripletMatrixViewBase<T, 1>
//{
//  protected:
//    // data
//    int* m_row_indices;
//    int* m_col_indices;
//    T*   m_values;
//
//    // matrix info
//    int m_rows = 0;
//    int m_cols = 0;
//
//    // triplet info
//    int m_triplet_index_offset = 0;
//    int m_triplet_count        = 0;
//    int m_total_triplet_count  = 0;
//
//  public:
//    MUDA_GENERIC TripletMatrixViewBase() = default;
//    MUDA_GENERIC TripletMatrixViewBase(int  rows,
//                                       int  cols,
//                                       int  triplet_index_offset,
//                                       int  triplet_count,
//                                       int  total_triplet_count,
//                                       int* row_indices,
//                                       int* col_indices,
//                                       T*   values)
//        : m_rows(rows)
//        , m_cols(cols)
//        , m_triplet_index_offset(triplet_index_offset)
//        , m_triplet_count(triplet_count)
//        , m_total_triplet_count(total_triplet_count)
//        , m_row_indices(row_indices)
//        , m_col_indices(col_indices)
//        , m_values(values)
//    {
//        MUDA_ASSERT(triplet_index_offset + triplet_count <= total_triplet_count,
//                    "TripletMatrixView: out of range, m_total_triplet_count=%d, "
//                    "your triplet_index_offset=%d, triplet_count=%d",
//                    total_triplet_count,
//                    triplet_index_offset,
//                    triplet_count);
//    }
//
//    MUDA_GENERIC auto block_rows() const { return m_rows; }
//    MUDA_GENERIC auto block_cols() const { return m_cols; }
//    MUDA_GENERIC auto triplet_count() const { return m_triplet_count; }
//    MUDA_GENERIC auto tripet_index_offset() const
//    {
//        return m_triplet_index_offset;
//    }
//    MUDA_GENERIC auto total_triplet_count() const
//    {
//        return m_total_triplet_count;
//    }
//
//    MUDA_GENERIC auto cviewer() const
//    {
//        return CTripletMatrixViewer<T, 1>{m_rows,
//                                          m_cols,
//                                          m_triplet_index_offset,
//                                          m_triplet_count,
//                                          m_total_triplet_count,
//                                          m_row_indices,
//                                          m_col_indices,
//                                          m_values};
//    }
//
//  protected:
//    MUDA_GENERIC TripletMatrixViewBase<T, 1> subview(int offset, int count) const
//    {
//        return TripletMatrixViewBase<T, 1>{m_rows,
//                                           m_cols,
//                                           m_triplet_index_offset + offset,
//                                           count,
//                                           m_total_triplet_count,
//                                           m_row_indices,
//                                           m_col_indices,
//                                           m_values};
//    }
//
//    MUDA_GENERIC TripletMatrixViewBase<T, 1> subview(int offset) const
//    {
//        MUDA_KERNEL_ASSERT(offset < m_triplet_count,
//                           "TripletMatrixView: offset is out of range, size=%d, your offset=%d",
//                           m_triplet_count,
//                           offset);
//        return subview(offset, m_triplet_count - offset);
//    }
//};
//
//template <typename T>
//class CTripletMatrixView<T, 1> : public TripletMatrixViewBase<T, 1>
//{
//    using Base = TripletMatrixViewBase<T, 1>;
//
//  public:
//    MUDA_GENERIC CTripletMatrixView(int        rows,
//                                    int        cols,
//                                    int        triplet_index_offset,
//                                    int        triplet_count,
//                                    int        total_triplet_count,
//                                    const int* row_indices,
//                                    const int* col_indices,
//                                    const T*   values)
//        : Base(rows,
//               cols,
//               triplet_index_offset,
//               triplet_count,
//               total_triplet_count,
//               const_cast<int*>(row_indices),
//               const_cast<int*>(col_indices),
//               const_cast<T*>(values))
//    {
//    }
//
//    MUDA_GENERIC CTripletMatrixView(const Base& base)
//        : Base(base)
//    {
//    }
//
//    MUDA_GENERIC CTripletMatrixView<T, 1> subview(int offset, int count) const
//    {
//        return CTripletMatrixView{Base::subview(offset, count)};
//    }
//
//    MUDA_GENERIC CTripletMatrixView<T, 1> subview(int offset) const
//    {
//        return CTripletMatrixView{Base::subview(offset)};
//    }
//
//    using Base::cviewer;
//};
//
//template <typename T>
//class TripletMatrixView<T, 1> : public TripletMatrixViewBase<T, 1>
//{
//    using Base = TripletMatrixViewBase<T, 1>;
//
//  public:
//    using Base::Base;
//    MUDA_GENERIC TripletMatrixView(const Base& base)
//        : Base(base)
//    {
//    }
//
//    MUDA_GENERIC TripletMatrixView(const CTripletMatrixView<T, 1>&) = delete;
//
//    MUDA_GENERIC TripletMatrixView<T, 1> subview(int offset, int count) const
//    {
//        return TripletMatrixView{Base::subview(offset, count)};
//    }
//
//    MUDA_GENERIC TripletMatrixView<T, 1> subview(int offset) const
//    {
//        return TripletMatrixView{Base::subview(offset)};
//    }
//
//    using Base::cviewer;
//    auto viewer() const
//    {
//        return TripletMatrixViewer<T, 1>{m_rows,
//                                         m_cols,
//                                         m_triplet_index_offset,
//                                         m_triplet_count,
//                                         m_total_triplet_count,
//                                         m_row_indices,
//                                         m_col_indices,
//                                         m_values};
//    }
//};
//}  // namespace muda
//
//namespace muda
//{
//template <typename T, int N>
//struct read_only_viewer<TripletMatrixView<T, N>>
//{
//    using type = CTripletMatrixView<T, N>;
//};
//
//template <typename T, int N>
//struct read_write_viewer<TripletMatrixView<T, N>>
//{
//    using type = TripletMatrixView<T, N>;
//};
//}  // namespace muda

#include <muda/view/view_base.h>
namespace muda
{
template <bool IsConst, typename Ty, int N>
class TripletMatrixViewBase
{
  public:
    static_assert(!std::is_const_v<Ty>, "Ty must be non-const");
    using ConstView                  = TripletMatrixViewBase<true, Ty, N>;
    using NonConstView               = TripletMatrixViewBase<false, Ty, N>;
    using ThisView                   = TripletMatrixViewBase<IsConst, Ty, N>;
    constexpr static bool IsConst    = IsConst;
    constexpr static bool IsNonConst = !IsConst;

  private:
    template <typename T>
    using auto_const_t = auto_const_t<IsConst, T>;
    template <typename T>
    using non_const_enable_t = std::enable_if_t<IsNonConst, T>;
    using CViewer            = CTripletMatrixViewer<Ty, N>;
    using Viewer             = TripletMatrixViewer<Ty, N>;
    using ThisViewer         = std::conditional_t<IsConst, CViewer, Viewer>;

  public:
    using BlockMatrix = Eigen::Matrix<Ty, N, N>;

  protected:
    // matrix info
    int m_block_rows = 0;
    int m_block_cols = 0;

    // triplet info
    int m_triplet_index_offset = 0;
    int m_triplet_count        = 0;
    int m_total_triplet_count  = 0;

    auto_const_t<int>*         m_block_row_indices = nullptr;
    auto_const_t<int>*         m_block_col_indices = nullptr;
    auto_const_t<BlockMatrix>* m_block_values      = nullptr;

  public:
    MUDA_GENERIC ThisView() = default;
    MUDA_GENERIC ThisView(int                        rows,
                          int                        cols,
                          int                        triplet_index_offset,
                          int                        triplet_count,
                          int                        total_triplet_count,
                          auto_const_t<int>*         block_row_indices,
                          auto_const_t<int>*         block_col_indices,
                          auto_const_t<BlockMatrix>* block_values)
        : m_block_rows(rows)
        , m_block_cols(cols)
        , m_triplet_index_offset(triplet_index_offset)
        , m_triplet_count(triplet_count)
        , m_total_triplet_count(total_triplet_count)
        , m_block_row_indices(block_row_indices)
        , m_block_col_indices(block_col_indices)
        , m_block_values(block_values)
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
        return ConstView{m_block_rows,
                         m_block_cols,
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
        return ThisView{m_block_rows,
                        m_block_cols,
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
        return CViewer{m_block_rows,
                       m_block_cols,
                       m_triplet_index_offset,
                       m_triplet_count,
                       m_total_triplet_count,
                       m_block_row_indices,
                       m_block_col_indices,
                       m_block_values};
    }

    MUDA_GENERIC auto viewer()
    {
        return ThisViewer{m_block_rows,
                          m_block_cols,
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


    // const access
    auto block_values() const { return m_block_values; }
    auto block_row_indices() const { return m_block_row_indices; }
    auto block_col_indices() const { return m_block_col_indices; }

    auto block_rows() const { return m_block_rows; }
    auto block_cols() const { return m_block_cols; }
    auto triplet_count() const { return m_triplet_count; }
    auto tripet_index_offset() const { return m_triplet_index_offset; }
    auto total_triplet_count() const { return m_total_triplet_count; }
};

template <bool IsConst, typename Ty>
class TripletMatrixViewBase<IsConst, Ty, 1>
{
  public:
    static_assert(!std::is_const_v<Ty>, "Ty must be non-const");
    using ConstView                  = TripletMatrixViewBase<true, Ty, 1>;
    using NonConstView               = TripletMatrixViewBase<false, Ty, 1>;
    using ThisView                   = TripletMatrixViewBase<IsConst, Ty, 1>;
    constexpr static bool IsConst    = IsConst;
    constexpr static bool IsNonConst = !IsConst;

  private:
    template <typename T>
    using auto_const_t = auto_const_t<IsConst, T>;
    template <typename T>
    using non_const_enable_t = std::enable_if_t<IsNonConst, T>;
    using CViewer            = CTripletMatrixViewer<Ty, 1>;
    using Viewer             = TripletMatrixViewer<Ty, 1>;
    using ThisViewer         = std::conditional_t<IsConst, CViewer, Viewer>;

  protected:
    // matrix info
    int m_rows = 0;
    int m_cols = 0;

    // triplet info
    int m_triplet_index_offset = 0;
    int m_triplet_count        = 0;
    int m_total_triplet_count  = 0;

    // data
    auto_const_t<int>* m_row_indices;
    auto_const_t<int>* m_col_indices;
    auto_const_t<Ty>*  m_values;


  public:
    MUDA_GENERIC ThisView() = default;

    MUDA_GENERIC ThisView(int                rows,
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
    auto_const_t<Ty>*  block_values() { return m_values; }
    auto_const_t<int>* block_row_indices() { return m_row_indices; }
    auto_const_t<int>* block_col_indices() { return m_col_indices; }


    // const access
    auto block_values() const { return m_values; }
    auto block_row_indices() const { return m_row_indices; }
    auto block_col_indices() const { return m_col_indices; }

    auto block_rows() const { return m_rows; }
    auto block_cols() const { return m_cols; }
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
