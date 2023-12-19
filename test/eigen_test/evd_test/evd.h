#pragma once
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/ext/eigen/evd.h>
#include "../eigen_test_common.h"

template <typename T, int N>
void evd_test()
{
    using namespace muda;
    using namespace Eigen;

    using Matrix = Eigen::Matrix<T, N, N>;
    using Vector = Eigen::Matrix<T, N, 1>;

    Matrix M, U;
    Vector eigen_values;
    M = Matrix::Identity();
    eigen::evd<T, N>(M, eigen_values, U);

    DeviceVar<Matrix> d_M = M;
    DeviceVar<Matrix> d_U = U;
    DeviceVar<Vector> d_eigen_values;

    Launch().apply(
        [M = d_M.viewer(), eigen_values = d_eigen_values.viewer(), U = d_U.viewer()] __device__() mutable
        {
            Matrix MM = *M;
            Matrix UU;
            eigen::evd<T, N>(MM, *eigen_values, UU);
        });
    Vector res_eigen_values = d_eigen_values;
    REQUIRE(approx_equal(eigen_values, res_eigen_values));
}

template <int N>
void evd_test()
{
    evd_test<float, N>();
    evd_test<double, N>();
}