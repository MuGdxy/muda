#pragma once
#include <cublas.h>
#include <cusparse.h>
#include "dense.h"
#include "sparse.h"
#include "dense_wrapper.h"

#include "../launch/launch_base.h"
#include "../check/checkCusparse.h"
#include "../check/checkCublas.h"

namespace muda
{
class blasContext
{
    cusparseHandle_t csh_;
    cublasHandle_t   cbh_;
    cudaStream_t     stream_;

  public:
    blasContext(cudaStream_t stream = nullptr)
        : stream_(stream)
    {
        checkCudaErrors(cusparseCreate(&csh_));
        checkCudaErrors(cublasCreate_v2(&cbh_));
        checkCudaErrors(cusparseSetStream(csh_, stream));
        checkCudaErrors(cublasSetStream_v2(cbh_, stream));
    }
    ~blasContext()
    {
        checkCudaErrors(cusparseDestroy(csh_));
        checkCudaErrors(cublasDestroy_v2(cbh_));
    }

    operator cusparseHandle_t() { return csh_; }
    operator cublasHandle_t() { return cbh_; }
    operator cudaStream_t() { return stream_; }

    cusparseHandle_t spHandle() { return csh_; };
    cublasHandle_t   dnHandle() { return cbh_; };
    cudaStream_t     streamHandle() { return stream_; };
};

template <typename T>
class pointerOrValue
{
    T v;

  public:
    pointerOrValue(T v)
        : v(v)
    {
    }
    auto data()
    {
        if constexpr(std::is_pointer_v<T>)
            return v;
        else
            return &v;
    }
};

class blas : public launch_base<blas>
{
    blasContext& ctx;

  public:
    blas(blasContext& ctx)
        : launch_base(ctx.streamHandle())
        , ctx(ctx)
    {
    }

    template <typename T, typename _ScalarValueOrPointer>
    blas& scal(_ScalarValueOrPointer alpha, dense::vec<T>& x_inout, int incx = 1)
    {
        auto a = pointerOrValue<T>(alpha);
        checkCudaErrors(
            dense::L1::cublasScal(ctx, x_inout.size(), a.data(), x_inout.data(), incx));
        return *this;
    }

    template <typename T, typename _ScalarValueOrPointer>
    blas& axpy(_ScalarValueOrPointer alpha,
               const dense::vec<T>&  x_in,
               dense::vec<T>&        y_inout,
               int                   incx = 1,
               int                   incy = 1)
    {
        if(x_in.size() != y_inout.size())
            throw std::runtime_error("vectors have different size");
        auto a = pointerOrValue<T>(alpha);
        checkCudaErrors(dense::L1::cublasAxpy(
            ctx, y_inout.size(), a.data(), x_in.data(), incx, y_inout.data(), incy));
        return *this;
    }

    template <typename T>
    blas& copy(const dense::vec<T>& x_in, dense::vec<T>& y_inout, int incx = 1, int incy = 1)
    {
        if(x_in.size() != y_inout.size())
            throw std::runtime_error("vectors have different size");
        checkCudaErrors(dense::L1::cublasCopy(
            ctx, y_inout.size(), x_in.data(), incx, y_inout.data(), incy));
        return *this;
    }

    template <typename T>
    blas& nrm2(dense::vec<T>& x_in, dense::L1::nrm2_result_t<T>& result, int incx = 1)
    {
        checkCudaErrors(dense::L1::cublasNrm2(ctx, x_in.size(), x_in.data(), incx, &result));
        return *this;
    }

    /// <summary>
    /// y = a * A * x + b * y
    /// </summary>
    /// <typeparam name="_Ty"></typeparam>
    /// <param name="alpha"></param>
    /// <param name="matA"></param>
    /// <param name="x_in"></param>
    /// <param name="beta"></param>
    /// <param name="y_inout"></param>
    /// <returns></returns>
    template <typename _Mat, typename _ScalarValueOrPointer>
    blas& spmv(_ScalarValueOrPointer                  alpha,
               sparse::TransposeView<_Mat>            matA,
               dense::vec<typename _Mat::value_type>& x_in,
               _ScalarValueOrPointer                  beta,
               dense::vec<typename _Mat::value_type>& y_inout,
               device_buffer<std::byte>&              external_buffer)
    {
        using value_type = typename _Mat::value_type;
        auto   a         = pointerOrValue<value_type>(alpha);
        auto   b         = pointerOrValue<value_type>(beta);
        size_t bufferSize;
        checkCudaErrors(cusparseSpMV_bufferSize(ctx,
                                                matA.trans_,
                                                a.data(),
                                                matA.mat_,
                                                x_in,
                                                b.data(),
                                                y_inout,
                                                details::cudaDataTypeMap_v<value_type>,
                                                cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                                                &bufferSize));

        details::set_stream_check(external_buffer, ctx);
        external_buffer.resize(bufferSize);

        checkCudaErrors(cusparseSpMV(ctx,
                                     matA.trans_,
                                     a.data(),
                                     matA.mat_,
                                     x_in,
                                     b.data(),
                                     y_inout,
                                     details::cudaDataTypeMap_v<value_type>,
                                     cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                                     external_buffer.data()));
        return *this;
    }

    template <typename _Mat, typename _ScalarValueOrPointer>
    blas& spmv(_ScalarValueOrPointer                  alpha,
               _Mat&                                  matA,
               dense::vec<typename _Mat::value_type>& x_in,
               _ScalarValueOrPointer                  beta,
               dense::vec<typename _Mat::value_type>& y_inout,
               device_buffer<std::byte>&              external_buffer)
    {
        auto m = sparse::TransposeView<_Mat>(matA);
        return spmv(alpha, m, x_in, beta, y_inout, external_buffer);
    }

    template <typename _Mat>
    blas& spmv(sparse::TransposeView<_Mat>            matA,
               dense::vec<typename _Mat::value_type>& x_in,
               dense::vec<typename _Mat::value_type>& y_inout,
               device_buffer<std::byte>&              external_buffer)
    {
        using value_type = typename _Mat::value_type;
        spmv(value_type(1), matA, x_in, value_type(0), y_inout, external_buffer);
        return *this;
    }

    template <typename _Mat>
    blas& spmv(_Mat&                                  matA,
               dense::vec<typename _Mat::value_type>& x_in,
               dense::vec<typename _Mat::value_type>& y_inout,
               device_buffer<std::byte>&              external_buffer)
    {
        using value_type = typename _Mat::value_type;
        spmv(value_type(1), matA, x_in, value_type(0), y_inout, external_buffer);
        return *this;
    }
};

inline blas on(blasContext& ctx)
{
    return blas(ctx);
}
}  // namespace muda