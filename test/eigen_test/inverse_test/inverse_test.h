#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/syntax_sugar.h>
#include <muda/ext/eigen/inverse.h>
#include <Eigen/Dense>
#include "../eigen_test_common.h"

using namespace muda;
using namespace Eigen;

template <typename T, int N, typename Algorithm = eigen::GaussEliminationInverse>
void inverse_test()
{
    using Matrix = Matrix<T, N, N>;
    using Vector = Vector<T, N>;

    Matrix A;

    auto ra = Vector::Random();
    A       = Matrix::Identity() + ra * ra.transpose();  // positive definite

    Matrix invA = A.inverse();

    DeviceVar<Matrix> d_result;
    DeviceVar<Matrix> d_A = A;


    Launch().apply([result = d_result.viewer(), A = d_A.viewer()] __device__() mutable
                   { result = eigen::inverse<T, N, Algorithm>(*A); });

    Matrix h_result = d_result;

    REQUIRE(approx_equal(invA, h_result));
}
