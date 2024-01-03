#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/ext/linear_system.h>
using namespace muda;
using namespace Eigen;
void linear_system_test()
{
    MatrixXf h_A = MatrixXf::Random(3, 3);
    VectorXf h_x = VectorXf::Random(3);
    VectorXf h_b = h_A * h_x;

    DeviceDenseMatrix<float> A = h_A;
    DeviceDenseVector<float> x = h_x;
    DeviceDenseVector<float> b;
    b.resize(h_b.size());

    LinearSystemContext ctx;
    ctx.mv<float>(A, x, b);

    VectorXf h_b2;
    b.copy_to(h_b2);

    REQUIRE(h_b2.isApprox(h_b));
}

TEST_CASE("linear_system_test", "[linear_system]")
{
    linear_system_test();
}
