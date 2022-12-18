#pragma once
#include <cublas.h>
#include <cusparse.h>
#pragma once
#include "../container/vector.h"
#include "../check/checkCusparse.h"
#include "../viewer.h"
#include "data_type_map.h"

namespace muda
{
namespace sparse
{
    template <typename _Mat>
    class TransposeView
    {
      public:
        using matrix_type = _Mat;
        TransposeView(matrix_type& mat)
            : mat_(mat)
            , trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
        {
        }
        TransposeView(matrix_type& mat, cusparseOperation_t trans_)
            : mat_(mat)
            , trans_(trans)
        {
        }
        matrix_type&        mat_;
        cusparseOperation_t trans_;
    };

    template <typename T>
    class sparse_matrix_base
    {
      protected:
        cusparseSpMatDescr_t spM_;

      public:
        using value_type = T;
        static constexpr cudaDataType_t dataType = details::cudaDataTypeMap_v<T>;

        sparse_matrix_base() = default;
        operator cusparseSpMatDescr_t() { return spM_; }
    };

    template <typename T, typename _RowOffsetType = int, typename _ColIndexType = int, cusparseIndexBase_t _IndexBase = CUSPARSE_INDEX_BASE_ZERO>
    class matCSR : public sparse_matrix_base<T>
    {
      public:
        using row_offset_type = _RowOffsetType;
        using col_index_type  = _ColIndexType;
        using this_type       = matCSR;

        matCSR(size_t           rows,
               size_t           cols,
               size_t           nNonZero,
               row_offset_type* d_rowOffsets,
               col_index_type*  d_colId,
               value_type*      d_values)
            : rows_(rows)
            , cols_(cols)
            , nNonZero_(nNonZero)
            , rowPtr_(d_rowOffsets)
            , colIdx_(d_colId)
            , values_(d_values)
        {
            checkCudaErrors(cusparseCreateCsr(&spM_,
                                              rows_,
                                              cols_,
                                              nNonZero_,
                                              d_rowOffsets,
                                              d_colId,
                                              d_values,
                                              details::cusparseIndexTypeMap_v<row_offset_type>,
                                              details::cusparseIndexTypeMap_v<col_index_type>,
                                              _IndexBase,
                                              dataType));
        }
        ~matCSR() { checkCudaErrors(cusparseDestroySpMat(spM_)); }
        operator cusparseSpMatDescr_t() { return spM_; }
        operator TransposeView<this_type>() { return TransposeView(*this); }

        row_offset_type* rowPtr() { return rowPtr_; }
        col_index_type*  colIdx() { return colIdx_; }
        value_type*      values() { return values_; };

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t nnz() const { return nNonZero_; }

      private:
        row_offset_type* rowPtr_;
        col_index_type*  colIdx_;
        value_type*      values_;

        cusparseSpMatDescr_t spM_;
        size_t               rows_;
        size_t               cols_;
        size_t               nNonZero_;
    };

    template <typename Mat>
    TransposeView<Mat> transpose(Mat& mat_)
    {
        return TransposeView(mat_, CUSPARSE_OPERATION_TRANSPOSE);
    }

    template <typename Mat>
    TransposeView<Mat> trans_conj(Mat& mat_)
    {
        return TransposeView(mat_, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
    }


    template <typename T, typename _RowOffsetType = int, typename _ColIndexType = int, cusparseIndexBase_t _IndexBase = CUSPARSE_INDEX_BASE_ZERO>
    class matrixBCSR
    {
        // TODO:
      public:
        using value_type      = T;
        using row_offset_type = _RowOffsetType;
        using col_index_type  = _ColIndexType;

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
}  // namespace sparse
}  // namespace muda

namespace muda
{
template <typename T>
inline __host__ auto make_csr(sparse::matCSR<T>& m)
{
    return csr<T>(m.rowPtr(), m.colIdx(), m.values(), m.rows(), m.cols(), m.nnz());
}

template <typename T>
inline __host__ auto make_viewer(sparse::matCSR<T>& m)
{
    return make_csr(m);
}
}  // namespace muda