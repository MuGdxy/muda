#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <iostream>
#include <muda/syntax_sugar.h>
#include <random>
#include "../example_common.h"
using namespace muda;
using namespace Eigen;
void mpm3d()
{
    example_desc(R"(This example we implement a simple mpm simulation.
The source code is from:
https://github.com/taichi-dev/taichi_benchmark/blob/main/suites/mpm/src/cuda/src/mpm3d.cu
we rewrite it using muda compute graph.)");

    auto n_particles = 8192;
    auto n_grid      = 128;
    auto dx          = 1.0f / n_grid;
    auto dt          = 2e-4f;

    auto p_rho   = 1.0f;
    auto p_vol   = std::pow(dx * 0.5, 2);
    auto p_mass  = p_vol * p_rho;
    auto gravity = 9.8f;
    auto bound   = 3;
    auto E       = 400.0f;

    auto x = DeviceBuffer<Vector2f>(n_particles);
    auto v = DeviceBuffer<Vector2f>(n_particles);
    auto C = DeviceBuffer<Matrix2f>(n_particles);
    auto J = DeviceBuffer<float>(n_particles);

    auto grid_v = DeviceBuffer2D<Vector2f>(Extent2D{n_grid, n_grid});
    auto grid_m = DeviceBuffer2D<float>(Extent2D{n_grid, n_grid});

    ComputeGraphVarManager manager;
    ComputeGraph           graph{manager};

    // init particles randomly



    graph.$node("init"){

    };
}

TEST_CASE("mpm3d", "[pba]")
{
    mpm3d();
}
