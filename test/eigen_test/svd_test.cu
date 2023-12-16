#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/syntax_sugar.h>
#include <muda/ext/eigen/svd.h>
#include "eigen_test_common.h"

using namespace muda;
using namespace Eigen;

// SVD
template <typename T, int N>
MUDA_HOST MUDA_DEVICE void SVD(const Eigen::Matrix<T, N, N>& F,
                               Eigen::Matrix<T, N, N>&       U,
                               Eigen::Matrix<T, N, N>&       S,
                               Eigen::Matrix<T, N, N>&       V)
{
    Eigen::JacobiSVD<Eigen::Matrix<T, N, N>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    S = svd.singularValues().asDiagonal();
    print("S = %f %f %f\n", S(0, 0), S(1, 1), S(2, 2));
}

void svd_failed()
{
    Matrix3f F, U, S, V;
    F << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    SVD<float, 3>(F, U, S, V);

    DeviceVar<Matrix3f> d_F = F;
    Launch()
        .apply(
            [d_F = d_F.viewer()] $()
            {
                Matrix3f F = d_F;
                Matrix3f U, S, V;
                S.setZero();
                SVD(F, U, S, V);
                print("S = %f %f %f", S(0, 0), S(1, 1), S(2, 2));
            })
        .wait();
    F = d_F;
    CHECK(approx_equal(F, S));
}

TEST_CASE("svd_failed", "[.svd_test_failed]")
{
    svd_failed();
}
template <typename T>
void svd_test()
{
    using Matrix = Eigen::Matrix<T, 3, 3>;
    using Vector = Eigen::Matrix<T, 3, 1>;

    Matrix F, U, V;
    Vector S;
    F = Matrix::Random();
    eigen::svd(F, U, S, V);

    DeviceVar<Matrix> d_F = F;
    DeviceVar<Vector> d_S;
    Launch()
        .apply(
            [F = d_F.viewer(), S = d_S.viewer()] $()
            {
                Matrix U, V;
                S->setZero();
                eigen::svd(*F, U, *S, V);
            })
        .wait();
    Vector res_S = d_S;
    CHECK(approx_equal(res_S, S));
}

TEST_CASE("svd_test", "[svd_test]")
{
    svd_test<float>();
}