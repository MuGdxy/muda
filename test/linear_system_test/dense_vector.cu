#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <Eigen/Dense>
#include <muda/ext/linear_system.h>
using namespace muda;
using namespace Eigen;

//using T                = float;
//constexpr int BlockDim = 3;

template <typename T>
void test_dense_vector(int dim)
{
    LinearSystemContext ctx;
    VectorX<T>          h_x   = VectorX<T>::Ones(dim);
    VectorX<T>          h_y   = VectorX<T>::Ones(dim);
    VectorX<T>          h_res = VectorX<T>::Zero(dim);

    DeviceDenseVector<T> x = h_x;
    DeviceDenseVector<T> y = h_y;
    DeviceDenseVector<T> z = h_res;

    // dot
    T res_dot = ctx.dot(x.cview(), y.cview());
    T gt_dot  = h_x.dot(h_y);

    REQUIRE(Approx(res_dot) == gt_dot);

    // norm
    T res_norm = ctx.norm(x.cview());
    T gt_norm  = h_x.norm();

    REQUIRE(Approx(res_norm) == gt_norm);

    // axpby
    T alpha = 2.0;
    T beta  = 3.0;
    ctx.axpby(alpha, x.cview(), beta, y.view());
    y.copy_to(h_res);
    VectorX<T> gt_axpby = alpha * h_x + beta * h_y;
    REQUIRE(h_res.isApprox(gt_axpby));

    // plus
    y = h_y;
    ctx.plus(x.cview(), y.cview(), z.view());
    z.copy_to(h_res);
    VectorX<T> gt_plus = h_x + h_y;
    REQUIRE(h_res.isApprox(gt_plus));
}

TEST_CASE("dense_vector", "[linear_system]")
{
    test_dense_vector<double>(1000);
    test_dense_vector<float>(1000);
}