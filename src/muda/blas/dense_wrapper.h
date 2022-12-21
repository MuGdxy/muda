#pragma once
#include <cublas.h>

namespace muda::dense::L1
{
template <typename T>
struct nrm2_result
{
};

template <typename T>
using nrm2_result_t = typename nrm2_result<T>::type;

template <>
struct nrm2_result<float>
{
    using type = float;
};
template <>
struct nrm2_result<double>
{
    using type = double;
};
template <>
struct nrm2_result<cuComplex>
{
    using type = float;
};
template <>
struct nrm2_result<cuDoubleComplex>
{
    using type = double;
};


// wrap cublasNrm2 for float/double/cuComplex/cuDoubleComplex
inline auto cublasNrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result)
{
    return cublasSnrm2_v2(handle, n, x, incx, result);
}
inline auto cublasNrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result)
{
    return cublasDnrm2_v2(handle, n, x, incx, result);
}
inline auto cublasNrm2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result)
{
    return cublasScnrm2_v2(handle, n, x, incx, result);
}
inline auto cublasNrm2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result)
{
    return cublasDznrm2_v2(handle, n, x, incx, result);
}


// wrap cublasDot for float/double/cuComplex/cuDoubleComplex
inline auto cublasDot(
    cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result)
{
    return cublasSdot_v2(handle, n, x, incx, y, incy, result);
}
inline auto cublasDot(
    cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result)
{
    return cublasDdot_v2(handle, n, x, incx, y, incy, result);
}
inline auto cublasDot(cublasHandle_t   handle,
                      int              n,
                      const cuComplex* x,
                      int              incx,
                      const cuComplex* y,
                      int              incy,
                      cuComplex*       result)
{
    return cublasCdotu_v2(handle, n, x, incx, y, incy, result);
}
inline auto cublasDot(cublasHandle_t         handle,
                      int                    n,
                      const cuDoubleComplex* x,
                      int                    incx,
                      const cuDoubleComplex* y,
                      int                    incy,
                      cuDoubleComplex*       result)
{
    return cublasZdotu_v2(handle, n, x, incx, y, incy, result);
}

// wrap cublasAxpy for float/double/cuComplex/cuDoubleComplex
inline auto cublasAxpy(
    cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
{
    return cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
}
inline auto cublasAxpy(
    cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy)
{
    return cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
}
inline auto cublasAxpy(cublasHandle_t   handle,
                       int              n,
                       const cuComplex* alpha,
                       const cuComplex* x,
                       int              incx,
                       cuComplex*       y,
                       int              incy)
{
    return cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy);
}
inline auto cublasAxpy(cublasHandle_t         handle,
                       int                    n,
                       const cuDoubleComplex* alpha,
                       const cuDoubleComplex* x,
                       int                    incx,
                       cuDoubleComplex*       y,
                       int                    incy)
{
    return cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy);
}

// wrap cublasScal for float/double/cuComplex/cuDoubleComplex
inline auto cublasScal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx)
{
    return cublasSscal_v2(handle, n, alpha, x, incx);
}
inline auto cublasScal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx)
{
    return cublasDscal_v2(handle, n, alpha, x, incx);
}
inline auto cublasScal(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx)
{
    return cublasCscal_v2(handle, n, alpha, x, incx);
}
inline auto cublasScal(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx)
{
    return cublasZscal_v2(handle, n, alpha, x, incx);
}

// wrap cublasCopy for float/double/cuComplex/cuDoubleComplex
inline auto cublasCopy(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
{
    return cublasScopy_v2(handle, n, x, incx, y, incy);
}
inline auto cublasCopy(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy)
{
    return cublasDcopy_v2(handle, n, x, incx, y, incy);
}
inline auto cublasCopy(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy)
{
    return cublasCcopy_v2(handle, n, x, incx, y, incy);
}
inline auto cublasCopy(cublasHandle_t         handle,
                       int                    n,
                       const cuDoubleComplex* x,
                       int                    incx,
                       cuDoubleComplex*       y,
                       int                    incy)
{
    return cublasZcopy_v2(handle, n, x, incx, y, incy);
}

// wrap cublasSwap for float/double/cuComplex/cuDoubleComplex
inline auto cublasSwap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
    return cublasSswap_v2(handle, n, x, incx, y, incy);
}
inline auto cublasSwap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy)
{
    return cublasDswap_v2(handle, n, x, incx, y, incy);
}
inline auto cublasSwap(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy)
{
    return cublasCswap_v2(handle, n, x, incx, y, incy);
}
inline auto cublasSwap(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy)
{
    return cublasZswap_v2(handle, n, x, incx, y, incy);
}

// wrap cublasIamax for float/double/cuComplex/cuDoubleComplex
inline auto cublasIamax(cublasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return cublasIsamax_v2(handle, n, x, incx, result);
}
inline auto cublasIamax(cublasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return cublasIdamax_v2(handle, n, x, incx, result);
}
inline auto cublasIamax(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result)
{
    return cublasIcamax_v2(handle, n, x, incx, result);
}
inline auto cublasIamax(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result)
{
    return cublasIzamax_v2(handle, n, x, incx, result);
}

// wrap cublasIamin for float/double/cuComplex/cuDoubleComplex
inline auto cublasIamin(cublasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return cublasIsamin_v2(handle, n, x, incx, result);
}
inline auto cublasIamin(cublasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return cublasIdamin_v2(handle, n, x, incx, result);
}
inline auto cublasIamin(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result)
{
    return cublasIcamin_v2(handle, n, x, incx, result);
}
inline auto cublasIamin(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result)
{
    return cublasIzamin_v2(handle, n, x, incx, result);
}

// wrap cublasasum for float/double/cuComplex/cuDoubleComplex
inline auto cublasAsum(cublasHandle_t handle, int n, const float* x, int incx, float* result)
{
    return cublasSasum_v2(handle, n, x, incx, result);
}
inline auto cublasAsum(cublasHandle_t handle, int n, const double* x, int incx, double* result)
{
    return cublasDasum_v2(handle, n, x, incx, result);
}
inline auto cublasAsum(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result)
{
    return cublasScasum_v2(handle, n, x, incx, result);
}
inline auto cublasAsum(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result)
{
    return cublasDzasum_v2(handle, n, x, incx, result);
}

//wrap cublasRot for float/double/cuComplex/cuDoubleComplex
inline auto cublasRot(
    cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s)
{
    return cublasSrot_v2(handle, n, x, incx, y, incy, c, s);
}
inline auto cublasRot(cublasHandle_t handle,
                      int            n,
                      double*        x,
                      int            incx,
                      double*        y,
                      int            incy,
                      const double*  c,
                      const double*  s)
{
    return cublasDrot_v2(handle, n, x, incx, y, incy, c, s);
}
inline auto cublasRot(cublasHandle_t handle,
                      int            n,
                      cuComplex*     x,
                      int            incx,
                      cuComplex*     y,
                      int            incy,
                      const float*   c,
                      const float*   s)
{
    return cublasCsrot_v2(handle, n, x, incx, y, incy, c, s);
}
inline auto cublasRot(cublasHandle_t   handle,
                      int              n,
                      cuDoubleComplex* x,
                      int              incx,
                      cuDoubleComplex* y,
                      int              incy,
                      const double*    c,
                      const double*    s)
{
    return cublasZdrot_v2(handle, n, x, incx, y, incy, c, s);
}

// wrap cublasRotg for float/double/cuComplex/cuDoubleComplex
inline auto cublasRotg(cublasHandle_t handle, float* a, float* b, float* c, float* s)
{
    return cublasSrotg_v2(handle, a, b, c, s);
}
inline auto cublasRotg(cublasHandle_t handle, double* a, double* b, double* c, double* s)
{
    return cublasDrotg_v2(handle, a, b, c, s);
}
inline auto cublasRotg(cublasHandle_t handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s)
{
    return cublasCrotg_v2(handle, a, b, c, s);
}
inline auto cublasRotg(cublasHandle_t   handle,
                       cuDoubleComplex* a,
                       cuDoubleComplex* b,
                       double*          c,
                       cuDoubleComplex* s)
{
    return cublasZrotg_v2(handle, a, b, c, s);
}

}  // namespace muda::dense::L1

namespace muda::dense::L2
{
// wrap cublas gemv for float/double/cuComplex/cuDoubleComplex
inline auto cublasGemv(cublasHandle_t    handle,
                       cublasOperation_t trans,
                       int               m,
                       int               n,
                       const float*      alpha, /* host or device pointer */
                       const float*      A,
                       int               lda,
                       const float*      x,
                       int               incx,
                       const float*      beta, /* host or device pointer */
                       float*            y,
                       int               incy)
{
    return cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
}  // namespace muda::dense::L2