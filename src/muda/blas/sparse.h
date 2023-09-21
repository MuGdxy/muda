#pragma once
#include <muda/tools/version.h>
#include <cublas.h>
#include <cusparse.h>
#include <muda/container/vector.h>
#include <muda/check/checkCusparse.h>
#include <muda/viewer.h>
#include "data_type_map.h"
#include "mat_view.h"

namespace muda
{
template <typename T>
class sparse_matrix_base
{
  protected:
    cusparseSpMatDescr_t m_spM;

  public:
    using value_type                         = T;
    static constexpr cudaDataType_t dataType = details::cudaDataTypeMap_v<T>;

    sparse_matrix_base() = default;
    operator cusparseSpMatDescr_t() { return m_spM; }
};

template <typename T, typename RowOffsetType = int, typename ColIndexType = int, cusparseIndexBase_t IndexBase = CUSPARSE_INDEX_BASE_ZERO>
class MatrixCSR : public sparse_matrix_base<T>
{
  public:
    using value_type      = T;
    using row_offset_type = RowOffsetType;
    using col_index_type  = ColIndexType;
    using this_type       = MatrixCSR;

    MatrixCSR(size_t           rows,
           size_t           cols,
           size_t           nNonZero,
           row_offset_type* d_rowOffsets,
           col_index_type*  d_colId,
           value_type*      d_values)
        : m_rows(rows)
        , m_cols(cols)
        , m_nNonZero(nNonZero)
        , m_rowPtr(d_rowOffsets)
        , m_colIdx(d_colId)
        , m_values(d_values)
    {
        checkCudaErrors(cusparseCreateCsr(&m_spM,
                                          m_rows,
                                          m_cols,
                                          m_nNonZero,
                                          d_rowOffsets,
                                          d_colId,
                                          d_values,
                                          details::cusparseIndexTypeMap_v<row_offset_type>,
                                          details::cusparseIndexTypeMap_v<col_index_type>,
                                          IndexBase,
                                          this->dataType));
    }
    ~MatrixCSR() { checkCudaErrors(cusparseDestroySpMat(m_spM)); }
    operator cusparseSpMatDescr_t() { return m_spM; }
    operator mat_view<this_type>() { return TransposeView(*this); }

    row_offset_type* rowPtr() { return m_rowPtr; }
    col_index_type*  colIdx() { return m_colIdx; }
    value_type*      values() { return m_values; };

    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    size_t nnz() const { return m_nNonZero; }

  private:
    row_offset_type* m_rowPtr;
    col_index_type*  m_colIdx;
    value_type*      m_values;

    cusparseSpMatDescr_t m_spM;
    size_t               m_rows;
    size_t               m_cols;
    size_t               m_nNonZero;
};

template <typename Mat>
mat_view<Mat> transpose(Mat& mat_)
{
    return mat_view(mat_, CUSPARSE_OPERATION_TRANSPOSE);
}

template <typename Mat>
mat_view<Mat> trans_conj(Mat& mat_)
{
    return mat_view(mat_, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
}


template <typename T, typename RowOffsetType = int, typename ColIndexType = int, cusparseIndexBase_t IndexBase = CUSPARSE_INDEX_BASE_ZERO>
class matrixBCSR
{
    // TODO:
  public:
    using value_type      = T;
    using row_offset_type = RowOffsetType;
    using col_index_type  = ColIndexType;

  public:
    matrixBCSR(size_t           rows,
               size_t           cols,
               size_t           nNonZero,
               row_offset_type* d_rowOffsets,
               col_index_type*  d_colId,
               value_type*      d_values)
        : rows_(rows)
        , cols_(cols)
        , nNonZero_(nNonZero)
        , rowOffsets_(d_rowOffsets)
        , colId_(d_colId)
        , values_(d_values)
    {
    }

    ~matrixBCSR() {}

  private:
    size_t           rows_;
    size_t           cols_;
    size_t           nNonZero_;
    row_offset_type* rowOffsets_;
    col_index_type*  colId_;
    value_type*      values_;
};
}  // namespace muda

namespace muda
{
template <typename T>
MUDA_INLINE MUDA_HOST auto make_csr(MatrixCSR<T>& m) MUDA_NOEXCEPT
{
    return CSRViewer<T>(m.rowPtr(), m.colIdx(), m.values(), m.rows(), m.cols(), m.nnz());
}

template <typename T>
MUDA_INLINE MUDA_HOST auto make_viewer(MatrixCSR<T>& m) MUDA_NOEXCEPT
{
    return make_csr(m);
}
}  // namespace muda
