#if MUDA_COMPUTE_GRAPH_ON
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <iostream>
#include <muda/syntax_sugar.h>
#include <random>
#include <muda/ext/eigen/svd.h>
#include <example_common.h>
using namespace muda;
using namespace Eigen;

void mpm3d()
{
    example_desc(R"(This example we implement a simple mpm simulation.
The source code is from:
https://github.com/taichi-dev/taichi_benchmark/blob/main/suites/mpm/src/cuda/src/mpm3d.cu
we rewrite it using muda compute graph.)");

    using Vector3   = Vector3f;
    using Matrix3x3 = Matrix3f;

    size_t dim        = 3;
    size_t n_grid     = 64;
    int    steps      = (2e-3 / 3e-4);
    float  dt         = 3e-4f;
    int    resolution = 500;

    int   n_particles = std::pow(n_grid, dim) / std::pow(2, dim - 1);
    float dx          = 1.0f / n_grid;
    float inv_dx      = n_grid;

    float p_vol  = std::pow(dx * 0.5, 2);
    float p_rho  = 1;
    float p_mass = p_vol * p_rho;

    Vector3 gravity{0, -9.8f, 0};
    float   bound_val = 3;
    float   E         = 400;   // Young's modulus
    float   nu        = 0.2f;  // Poisson's ratio
    // Lame parameters
    float mu_0     = E / (2 * (1 + nu));
    float lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

    int group_size = n_particles / 3;

    // particle data
    auto x   = DeviceBuffer<Vector3>(n_particles);    // position
    auto v   = DeviceBuffer<Vector3>(n_particles);    // velocity
    auto C   = DeviceBuffer<Matrix3x3>(n_particles);  // affine velocity field
    auto F   = DeviceBuffer<Matrix3x3>(n_particles);  // deformation gradient
    auto Jp  = DeviceBuffer<float>(n_particles);      // plastic deformation
    auto mat = DeviceBuffer<int>(n_particles);        // material id

    auto grid_v = DeviceBuffer3D<Vector3>(Extent3D{n_grid, n_grid, n_grid});  // grid node momentum/velocity)
    auto grid_m = DeviceBuffer3D<float>(Extent3D{n_grid, n_grid, n_grid});  // grid node mass

    ComputeGraphVarManager manager;

    auto& x_var   = manager.create_var("x", x.view());
    auto& v_var   = manager.create_var("v", v.view());
    auto& C_var   = manager.create_var("C", C.view());
    auto& F_var   = manager.create_var("F", F.view());
    auto& Jp_var  = manager.create_var("Jp", Jp.view());
    auto& mat_var = manager.create_var("mat", mat.view());

    auto& grid_v_var = manager.create_var("grid_v", grid_v.view());
    auto& grid_m_var = manager.create_var("grid_m", grid_m.view());

    ComputeGraph graph{manager};

    // init particles randomly
    graph.$node("reset_grid")
    {
        ParallelFor(n_grid)  //
            .apply(grid_v.total_size(),
                   [grid_v = grid_v_var.viewer(), grid_m = grid_m_var.viewer()] $(int i)
                   {
                       grid_v.flatten(i) = Vector3::Zero();
                       grid_m.flatten(i) = 0;
                   });
    };

    graph.$node("p2g")
    {

        ParallelFor(256).kernel_name("p2g").apply(
            x.size(),
            [x   = x_var.viewer(),
             F   = F_var.viewer(),
             C   = C_var.cviewer(),
             Jp  = Jp_var.cviewer(),
             mat = mat_var.cviewer(),
             dt,
             mu_0,
             lambda_0] $(int p)
            {
                auto    Xp   = x(p);
                Vector3 base = Xp - 0.5f * Vector3::Ones();
                Vector3 fx   = Xp - base;
                // w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                // Quadratic kernels
                Vector3 f1  = 0.5f * (1.5f - fx.array()).pow(2);
                Vector3 f2  = 0.75f - (fx.array() - 1).pow(2);
                Vector3 f3  = 0.5f * (fx.array() - 0.5f).pow(2);
                Vector3 w[] = {f1, f2, f3};
                F(p)        = (Matrix3x3::Identity() + dt * C(p)) * F(p);

                // hardening coefficient: snow->water
                auto h = exp(10.0f * (1.0f - Jp(p)));
                if(mat(p) == 1)
                    h = 0.3f;  // jelly, make it softer
                auto mu = mu_0 * h;
                auto lambda = lambda_0 * h;  // lame parameters, controls the deformation

                if(mat(p) == 0)  // liquid
                {
                    mu = 0.0f;
                };

                Vector3 sig;
                Matrix3x3 U, V;
                eigen::svd(F(p), U, sig, V);
            });
    };
}

TEST_CASE("mpm3d", "[pba]")
{
    mpm3d();
}
#endif