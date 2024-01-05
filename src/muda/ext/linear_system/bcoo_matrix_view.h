#pragma once
#include <muda/ext/linear_system/triplet_matrix_view.h>
namespace muda
{
template <typename T, int N>
using BCOOMatrixView = TripletMatrixView<T, N>;
template <typename T, int N>
using CBCOOMatrixView = CTripletMatrixView<T, N>;

template <typename Ty>
class COOMatrixViewBase
{
  protected:
    // data
    int* m_row_indices;
    int* m_col_indices;
    Ty*  m_values;

    // matrix info
    int m_rows = 0;
    int m_cols = 0;

    // triplet info
    int                  m_triplet_index_offset = 0;
    int                  m_triplet_count        = 0;
    int                  m_total_triplet_count  = 0;
    cusparseMatDescr_t   m_legacy_descr         = nullptr;
    cusparseSpMatDescr_t m_descr                = nullptr;
    bool                 m_trans                = false;

  public:
    MUDA_GENERIC COOMatrixViewBase() = default;
    MUDA_GENERIC COOMatrixViewBase(int                  rows,
                                   int                  cols,
                                   int                  triplet_index_offset,
                                   int                  triplet_count,
                                   int                  total_triplet_count,
                                   int*                 row_indices,
                                   int*                 col_indices,
                                   Ty*                  values,
                                   cusparseMatDescr_t   legacy_descr,
                                   cusparseSpMatDescr_t descr,
                                   bool                 trans)

        : m_rows(rows)
        , m_cols(cols)
        , m_triplet_index_offset(triplet_index_offset)
        , m_triplet_count(triplet_count)
        , m_total_triplet_count(total_triplet_count)
        , m_row_indices(row_indices)
        , m_col_indices(col_indices)
        , m_values(values)
        , m_legacy_descr(legacy_descr)
        , m_descr(descr)
        , m_trans(trans)
    {
        MUDA_ASSERT(triplet_index_offset + triplet_count <= total_triplet_count,
                    "TripletMatrixView: out of range, m_total_triplet_count=%d, "
                    "your triplet_index_offset=%d, triplet_count=%d",
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

    MUDA_GENERIC auto row_indices() const { return m_row_indices; }
    MUDA_GENERIC auto col_indices() const { return m_col_indices; }
    MUDA_GENERIC auto values() const { return m_values; }

    MUDA_GENERIC auto legacy_descr() const { return m_legacy_descr; }
    MUDA_GENERIC auto descr() const { return m_descr; }


    MUDA_GENERIC auto cviewer() const
    {
        MUDA_ASSERT(!m_trans,
                    "COOMatrixView: viewer() is not supported for "
                    "transposed matrix, please use a non-transposed view of this matrix");
        return CTripletMatrixViewer<Ty, 1>{m_rows,
                                           m_cols,
                                           m_triplet_index_offset,
                                           m_triplet_count,
                                           m_total_triplet_count,
                                           m_row_indices,
                                           m_col_indices,
                                           m_values};
    }

  protected:
    MUDA_GENERIC auto T() const
    {
        return COOMatrixViewBase{m_rows,
                                 m_cols,
                                 m_triplet_index_offset,
                                 m_triplet_count,
                                 m_total_triplet_count,
                                 m_col_indices,
                                 m_row_indices,
                                 m_values,
                                 m_legacy_descr,
                                 m_descr,
                                 !m_trans};
    }
};

template <typename Ty>
class CCOOMatrixView : public COOMatrixViewBase<Ty>
{
    using Base = COOMatrixViewBase<Ty>;

  public:
    using Base::Base;
    MUDA_GENERIC CCOOMatrixView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CCOOMatrixView(int                  rows,
                                int                  cols,
                                int                  triplet_index_offset,
                                int                  triplet_count,
                                int                  total_triplet_count,
                                const int*           row_indices,
                                const int*           col_indices,
                                const Ty*            values,
                                cusparseMatDescr_t   legacy_descr,
                                cusparseSpMatDescr_t descr,
                                bool                 trans)
        : Base(rows,
               cols,
               triplet_index_offset,
               triplet_count,
               total_triplet_count,
               const_cast<int*>(row_indices),
               const_cast<int*>(col_indices),
               const_cast<Ty*>(values),
               legacy_descr,
               descr,
               trans)
    {
    }

    using Base::cviewer;

    auto T() const { return CCOOMatrixView{Base::T()}; }
};


template <typename Ty>
class COOMatrixView : public COOMatrixViewBase<Ty>
{
    using Base = COOMatrixViewBase<Ty>;

  public:
    using Base::Base;

    MUDA_GENERIC COOMatrixView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC COOMatrixView(const CCOOMatrixView<Ty>&) = delete;

    using Base::cviewer;

    MUDA_GENERIC auto T() const { return CCOOMatrixView{Base::T()}; }

    MUDA_GENERIC auto viewer() const
    {
        return TripletMatrixViewer<Ty, 1>{m_rows,
                                          m_cols,
                                          m_triplet_index_offset,
                                          m_triplet_count,
                                          m_total_triplet_count,
                                          m_row_indices,
                                          m_col_indices,
                                          m_values};
    }
};


}  // namespace muda

namespace muda
{
template <typename T, int N>
struct read_only_viewer<BCOOMatrixView<T, N>>
{
    using type = CBCOOMatrixView<T, N>;
};

template <typename T, int N>
struct read_write_viewer<CBCOOMatrixView<T, N>>
{
    using type = BCOOMatrixView<T, N>;
};
}  // namespace muda
#include "details/bcoo_matrix_view.inl"
