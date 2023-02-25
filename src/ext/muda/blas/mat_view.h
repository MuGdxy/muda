#pragma once
#include <cublas.h>
#include <cusparse.h>

namespace muda
{
template <typename Mat>
class mat_view
{
  public:
    using matrix_type = Mat;
    using value_type  = typename matrix_type::value_type;
    
    mat_view(matrix_type& mat)
        : m_mat(mat)
        , m_trans(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
    }
    
    mat_view(matrix_type& mat, cusparseOperation_t trans)
        : m_mat(mat)
        , m_trans(trans)
    {
    }
    
    matrix_type&        m_mat;
    cusparseOperation_t m_trans;
};
}  // namespace muda