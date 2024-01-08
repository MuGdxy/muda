#pragma once
#include <muda/viewer/viewer_base.h>
#include <muda/viewer/details/csr_check.inl>

namespace muda
{
/// <summary>
/// a const viewer that allows to access a csr matrix
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class CCSRViewer : public ViewerBase<false> // TODO
{
    MUDA_VIEWER_COMMON_NAME(CCSRViewer);

  public:
    class CElem
    {
        int                  m_row;
        int                  m_col;
        int                  m_global_offset;
        const CCSRViewer<T>& csr_;
        MUDA_GENERIC CElem(const CCSRViewer<T>& csr, int row, int col, int global_offset) MUDA_NOEXCEPT
            : csr_(csr),
              m_row(row),
              m_col(col),
              m_global_offset(global_offset)
        {
        }

      public:
        friend class CCSRViewer<T>;
        //trivial copy constructor
        MUDA_GENERIC CElem(const CElem& e) = default;
        //trivial copy assignment
        MUDA_GENERIC CElem& operator=(const CElem& e) = default;
        MUDA_GENERIC operator const T&() const MUDA_NOEXCEPT
        {
            return csr_.m_values[m_global_offset];
        }
        int2 pos() const MUDA_NOEXCEPT { return make_int2(m_row, m_col); }
        int  global_offset() const MUDA_NOEXCEPT { return m_global_offset; }
    };

    MUDA_GENERIC CCSRViewer() MUDA_NOEXCEPT : m_values(nullptr),
                                              m_colIdx(nullptr),
                                              m_rowPtr(nullptr),
                                              m_nnz(0),
                                              m_rows(0),
                                              m_cols(0)
    {
    }

    MUDA_GENERIC CCSRViewer(const int* rowPtr,
                            const int* colIdx,
                            const T*   values,
                            int        rows,
                            int        cols,
                            int nNonZeros) MUDA_NOEXCEPT : m_rowPtr(rowPtr),
                                                           m_colIdx(colIdx),
                                                           m_values(values),
                                                           m_nnz(nNonZeros),
                                                           m_rows(rows),
                                                           m_cols(cols)
    {
    }

    // rows getter
    MUDA_GENERIC int rows() const MUDA_NOEXCEPT { return m_rows; }
    // cols getter
    MUDA_GENERIC int cols() const MUDA_NOEXCEPT { return m_cols; }
    // nnz getter
    MUDA_GENERIC int nnz() const MUDA_NOEXCEPT { return m_nnz; }

    // get by row and col as if it is a dense matrix
    MUDA_GENERIC T operator()(int row, int col) const MUDA_NOEXCEPT
    {
        check_range(row, col);
        for(int i = m_rowPtr[row]; i < m_rowPtr[row + 1]; i++)
        {
            if(m_colIdx[i] == col)
                return m_values[i];
        }
        return 0;
    }

    // read-only element
    MUDA_GENERIC CElem ro_elem(int row, int local_offset) const MUDA_NOEXCEPT
    {
        int global_offset;
        check_all(row, local_offset, global_offset);
        return CElem(*this, row, m_colIdx[global_offset], global_offset);
    }

    MUDA_GENERIC int nnz(int row) const MUDA_NOEXCEPT
    {
        check_row(row);
        return m_rowPtr[row + 1] - m_rowPtr[row];
    }

  private:
    const int* m_rowPtr;
    const int* m_colIdx;
    const T*   m_values;
    int        m_nnz;
    int        m_rows;
    int        m_cols;
    MUDA_INLINE MUDA_GENERIC void check_range(int row, int col) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::csr_check_range(row, col, m_rows, m_cols, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_row(int row) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::csr_check_row(row, m_rows, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_local_offset(int row, int offset) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::csr_check_local_offset(
                row, offset, m_rows, m_rowPtr, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_global_offset(int globalOffset) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::csr_check_global_offset(
                globalOffset, m_nnz, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_all(int row, int local_offset, int& global_offset) const MUDA_NOEXCEPT
    {
        check_row(row);
        check_local_offset(row, local_offset);
        global_offset = m_rowPtr[row] + local_offset;
        check_global_offset(global_offset);
    }
};


/// <summary>
/// a viwer that allows to access a CSR sparse matrix
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class CSRViewer : public ViewerBase
{
    MUDA_VIEWER_COMMON_NAME(CSRViewer);

  public:
    class Elem
    {
        int           m_row;
        int           m_col;
        int           m_global_offset;
        CSRViewer<T>& csr_;
        MUDA_GENERIC Elem(CSRViewer<T>& csr, int row, int col, int global_offset) MUDA_NOEXCEPT
            : csr_(csr),
              m_row(row),
              m_col(col),
              m_global_offset(global_offset)
        {
        }

      public:
        friend class CSRViewer<T>;
        //trivial copy constructor
        MUDA_GENERIC Elem(const Elem& e) = default;
        //trivial copy assignment
        MUDA_GENERIC Elem& operator=(const Elem& e) = default;
        MUDA_GENERIC operator const T&() const MUDA_NOEXCEPT
        {
            return csr_.m_values[m_global_offset];
        }
        MUDA_GENERIC operator T&() MUDA_NOEXCEPT
        {
            return csr_.m_values[m_global_offset];
        }
        int2 pos() const MUDA_NOEXCEPT { return make_int2(m_row, m_col); }
        int  global_offset() const MUDA_NOEXCEPT { return m_global_offset; }

        MUDA_GENERIC T& operator=(const T& v) MUDA_NOEXCEPT
        {
            auto& pos = csr_.m_values[m_global_offset];
            pos       = v;
            return pos;
        }
    };
    class CElem
    {
        int                 m_row;
        int                 m_col;
        int                 m_global_offset;
        const CSRViewer<T>& csr_;
        MUDA_GENERIC CElem(const CSRViewer<T>& csr, int row, int col, int global_offset) MUDA_NOEXCEPT
            : csr_(csr),
              m_row(row),
              m_col(col),
              m_global_offset(global_offset)
        {
        }

      public:
        friend class CSRViewer<T>;
        //trivial copy constructor
        MUDA_GENERIC CElem(const CElem& e) = default;
        //trivial copy assignment
        MUDA_GENERIC CElem& operator=(const CElem& e) = default;
        MUDA_GENERIC operator const T&() const MUDA_NOEXCEPT
        {
            return csr_.m_values[m_global_offset];
        }
        int2 pos() const MUDA_NOEXCEPT { return make_int2(m_row, m_col); }
        int  global_offset() const MUDA_NOEXCEPT { return m_global_offset; }
    };

    MUDA_GENERIC CSRViewer() MUDA_NOEXCEPT : m_values(nullptr),
                                             m_colIdx(nullptr),
                                             m_rowPtr(nullptr),
                                             m_nnz(0),
                                             m_rows(0),
                                             m_cols(0)
    {
    }

    MUDA_GENERIC CSRViewer(int* rowPtr, int* colIdx, T* values, int rows, int cols, int nNonZeros) MUDA_NOEXCEPT
        : m_rowPtr(rowPtr),
          m_colIdx(colIdx),
          m_values(values),
          m_nnz(nNonZeros),
          m_rows(rows),
          m_cols(cols)
    {
    }

    // rows getter
    MUDA_GENERIC int rows() const MUDA_NOEXCEPT { return m_rows; }
    // cols getter
    MUDA_GENERIC int cols() const MUDA_NOEXCEPT { return m_cols; }
    // nnz getter
    MUDA_GENERIC int nnz() const MUDA_NOEXCEPT { return m_nnz; }

    // get by row and col as if it is a dense matrix
    MUDA_GENERIC T operator()(int row, int col) const MUDA_NOEXCEPT
    {
        check_range(row, col);
        for(int i = m_rowPtr[row]; i < m_rowPtr[row + 1]; i++)
        {
            if(m_colIdx[i] == col)
                return m_values[i];
        }
        return 0;
    }
    // read-write element
    MUDA_GENERIC Elem rw_elem(int row, int local_offset) MUDA_NOEXCEPT
    {
        int global_offset;
        check_all(row, local_offset, global_offset);
        return Elem(*this, row, m_colIdx[global_offset], global_offset);
    }
    // read-only element
    MUDA_GENERIC CElem ro_elem(int row, int local_offset) const MUDA_NOEXCEPT
    {
        int global_offset;
        check_all(row, local_offset, global_offset);
        return CElem(*this, row, m_colIdx[global_offset], global_offset);
    }

    MUDA_GENERIC void place_row(int row, int global_offset) MUDA_NOEXCEPT
    {
        check_row(row);
        m_rowPtr[row] = global_offset;
    }

    MUDA_GENERIC void place_tail() MUDA_NOEXCEPT { m_rowPtr[m_rows] = m_nnz; }

    MUDA_GENERIC int place_col(int row, int local_offset, int col) MUDA_NOEXCEPT
    {
        check_row(row);
        int global_offset = m_rowPtr[row] + local_offset;
        check_global_offset(global_offset);
        m_colIdx[global_offset] = col;
        return global_offset;
    }

    MUDA_GENERIC int place_col(int row, int local_offset, int col, const T& v) MUDA_NOEXCEPT
    {
        check_row(row);
        int global_offset = m_rowPtr[row] + local_offset;
        check_global_offset(global_offset);
        m_colIdx[global_offset] = col;
        m_values[global_offset] = v;
        return global_offset;
    }

    MUDA_GENERIC int place_col(int global_offset, int col, const T& v) MUDA_NOEXCEPT
    {
        check_global_offset(global_offset);
        m_colIdx[global_offset] = col;
        m_values[global_offset] = v;
        return global_offset;
    }

    MUDA_GENERIC int nnz(int row) const MUDA_NOEXCEPT
    {
        check_row(row);
        return m_rowPtr[row + 1] - m_rowPtr[row];
    }

  private:
    int* m_rowPtr;
    int* m_colIdx;
    T*   m_values;
    int  m_nnz;
    int  m_rows;
    int  m_cols;
    MUDA_INLINE MUDA_GENERIC void check_range(int row, int col) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::csr_check_range(row, col, m_rows, m_cols, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_row(int row) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::csr_check_row(row, m_rows, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_local_offset(int row, int offset) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::csr_check_local_offset(
                row, offset, m_rows, m_rowPtr, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_global_offset(int globalOffset) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::csr_check_global_offset(
                globalOffset, m_nnz, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_all(int row, int local_offset, int& global_offset) const MUDA_NOEXCEPT
    {
        check_row(row);
        check_local_offset(row, local_offset);
        global_offset = m_rowPtr[row] + local_offset;
        check_global_offset(global_offset);
    }
};

template <typename T>
struct read_only_viewer<CSRViewer<T>>
{
    using type = CCSRViewer<T>;
};

template <typename T>
struct read_write_viewer<CCSRViewer<T>>
{
    using type = CSRViewer<T>;
};
}  // namespace muda
