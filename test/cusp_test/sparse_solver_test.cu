#include <catch2/catch.hpp>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/monitor.h>
#include <cusp/linear_operator.h>

void cg_test()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::gallery::poisson5pt(A, 10, 10);
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0.0f);
    cusp::monitor<float>                      monitor(x, 20, 1e-4);
    cusp::identity_operator<float, cusp::device_memory> M(A.num_rows, A.num_cols);
    cusp::krylov::cg(A, x, x, monitor);
}


TEST_CASE("cg", "[cusp]")
{
    cg_test();
}
