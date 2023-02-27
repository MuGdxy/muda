#pragma once
#include <muda/tools/version.h>

#include <muda/check/checkCusparse.h>
#include <muda/check/checkCublas.h>
#include <muda/check/checkCudaErrors.h>
#include <muda/launch/launch_base.h>

#include <cublas.h>
#include <cusparse.h>

#include "mat_view.h"
#include "dense.h"
#include "sparse.h"
#include "dense_wrapper.h"

namespace muda
{
class blasContext
{
    cusparseHandle_t m_csh;
    cublasHandle_t   m_cbh;
    cudaStream_t     m_stream;

  public:
    blasContext(cudaStream_t stream = nullptr)
        : m_stream(stream)
    {
        checkCudaErrors(cublasCreate_v2(&m_cbh));
        checkCudaErrors(cublasSetStream_v2(m_cbh, stream));

        checkCudaErrors(cusparseCreate(&m_csh));
        checkCudaErrors(cusparseSetStream(m_csh, stream));
    }

    ~blasContext()
    {
        checkCudaErrors(cublasDestroy_v2(m_cbh));

        checkCudaErrors(cusparseDestroy(m_csh));
    }

    operator cusparseHandle_t() { return m_csh; }
    cusparseHandle_t spHandle() { return m_csh; };

    operator cublasHandle_t() { return m_cbh; }
    operator cudaStream_t() { return m_stream; }
    cublasHandle_t   dnHandle() { return m_cbh; };
    cudaStream_t     stream() { return m_stream; };
};

template <typename T>
class ptr_or_val
{
    T m_v;

  public:
    ptr_or_val(T v)
        : m_v(v)
    {
    }
    auto data()
    {
        if constexpr(std::is_pointer_v<T>)
            return m_v;
        else
            return &m_v;
    }
};

class blas : public launch_base<blas>
{
    blasContext& m_ctx;

  public:
    blas(blasContext& ctx)
        : launch_base(ctx.stream())
        , m_ctx(ctx)
    {
    }

    template <typename T, typename ScalarValueOrPointer>
    blas& scal(ScalarValueOrPointer alpha, dense_vec<T>& x_inout, int incx = 1)
    {
        auto a = ptr_or_val<T>(alpha);
        checkCudaErrors(cublas::L1::cublasScal(
            m_ctx, x_inout.size(), a.data(), x_inout.data(), incx));
        return *this;
    }

    template <typename T, typename ScalarValueOrPointer>
    blas& axpy(ScalarValueOrPointer alpha,
               const dense_vec<T>&  x_in,
               dense_vec<T>&        y_inout,
               int                  incx = 1,
               int                  incy = 1)
    {
        if(x_in.size() != y_inout.size())
            throw std::runtime_error("vectors have different size");
        auto a = ptr_or_val<T>(alpha);
        checkCudaErrors(cublas::L1::cublasAxpy(
            m_ctx, y_inout.size(), a.data(), x_in.data(), incx, y_inout.data(), incy));
        return *this;
    }

    template <typename T>
    blas& copy(const dense_vec<T>& x_in, dense_vec<T>& y_inout, int incx = 1, int incy = 1)
    {
        if(x_in.size() != y_inout.size())
            throw std::runtime_error("vectors have different size");
        checkCudaErrors(cublas::L1::cublasCopy(
            m_ctx, y_inout.size(), x_in.data(), incx, y_inout.data(), incy));
        return *this;
    }

    template <typename T>
    blas& nrm2(dense_vec<T>& x_in, cublas::L1::nrm2_result_t<T>& result, int incx = 1)
    {
        checkCudaErrors(cublas::L1::cublasNrm2(m_ctx, x_in.size(), x_in.data(), incx, &result));
        return *this;
    }

    /// <summary>
    /// y = a * A * x + b * y
    /// </summary>
    template <typename MatView, typename ScalarValueOrPointer>
    blas& spmv(ScalarValueOrPointer                                 alpha,
               MatView&&                                            matA,
               dense_vec<typename raw_type_t<MatView>::value_type>& x_in,
               ScalarValueOrPointer                                 beta,
               dense_vec<typename raw_type_t<MatView>::value_type>& y_inout,
               device_buffer<std::byte>& external_buffer,
               cusparseSpMVAlg_t         alg = static_cast<cusparseSpMVAlg_t>(0))
    {


        using value_type = typename raw_type_t<MatView>::value_type;

        auto m = mat_view(std::forward<MatView>(matA));
        auto a = ptr_or_val<value_type>(alpha);
        auto b = ptr_or_val<value_type>(beta);

        size_t bufferSize;

        checkCudaErrors(cusparseSpMV_bufferSize(m_ctx,
                                                m.m_trans,
                                                a.data(),
                                                m.m_mat,
                                                x_in,
                                                b.data(),
                                                y_inout,
                                                details::cudaDataTypeMap_v<value_type>,
                                                alg,
                                                &bufferSize));

        details::set_stream_check(external_buffer, m_ctx);
        external_buffer.resize(bufferSize);

        checkCudaErrors(cusparseSpMV(m_ctx,
                                     m.m_trans,
                                     a.data(),
                                     m.m_mat,
                                     x_in,
                                     b.data(),
                                     y_inout,
                                     details::cudaDataTypeMap_v<value_type>,
                                     alg,
                                     external_buffer.data()));
        return *this;
    }

    //template <typename Mat, typename ScalarValueOrPointer>
    //blas& spmv(ScalarValueOrPointer                 alpha,
    //           Mat&                                 matA,
    //           dense_vec<typename Mat::value_type>& x_in,
    //           ScalarValueOrPointer                 beta,
    //           dense_vec<typename Mat::value_type>& y_inout,
    //           device_buffer<std::byte>&            external_buffer)
    //{
    //    auto m = mat_view<Mat>(matA);
    //    return spmv(alpha, m, x_in, beta, y_inout, external_buffer);
    //}

    //template <typename Mat>
    //blas& spmv(mat_view<Mat>                        matA,
    //           dense_vec<typename Mat::value_type>& x_in,
    //           dense_vec<typename Mat::value_type>& y_inout,
    //           device_buffer<std::byte>&            external_buffer)
    //{
    //    using value_type = typename Mat::value_type;
    //    spmv(value_type(1), matA, x_in, value_type(0), y_inout, external_buffer);
    //    return *this;
    //}

    template <typename MatView>
    blas& spmv(MatView&&                                            matA,
               dense_vec<typename raw_type_t<MatView>::value_type>& x_in,
               dense_vec<typename raw_type_t<MatView>::value_type>& y_inout,
               device_buffer<std::byte>& external_buffer,
               cusparseSpMVAlg_t         alg = static_cast<cusparseSpMVAlg_t>(0))
    {
        using value_type = typename raw_type_t<MatView>::value_type;
        spmv(value_type(1), std::forward<MatView>(matA), x_in, value_type(0), y_inout, external_buffer, alg);
        return *this;
    }
};

inline blas on(blasContext& ctx)
{
    return blas(ctx);
}
}  // namespace muda