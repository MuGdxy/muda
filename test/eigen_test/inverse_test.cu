#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/syntax_sugar.h>
#include <muda/ext/eigen/svd.h>
#include "eigen_test_common.h"

using namespace muda;
using namespace Eigen;


using T         = float;
constexpr int N = 3;

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
    Matrix h_result = d_result;

    Launch().apply([result = d_result.viewer(), A = d_A.viewer()] __device__() mutable
                   { result = A->inverse(); });

    REQUIRE(approx_equal(invA, h_result));
}


TEST_CASE("inverse_test", "[svd_test]") 
{
    
}

TEST_CASE("inverse_failed", "[.inverse_failed]")
{
    inverse_test();
}
