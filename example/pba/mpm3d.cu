//#include <catch2/catch.hpp>
//#include <muda/muda.h>
//#include <Eigen/Dense>
//#include <memory>
//#include <chrono>
//#include <iostream>
//#include "../example_common.h"
//using namespace muda;
//#undef min
//#undef max
////
//// Based on https://github.com/Aisk1436/mpm3d
////
//
//// Benchmark MPM3D
//// dim, steps, dt = 3, 25, 8e-5
//
//using Vector  = Eigen::Vector3f;
//using Matrix  = Eigen::Matrix3f;
//using Vectori = Eigen::Vector3i;
//using Real    = float;
//
//// TODO global var
//__constant__ constexpr Real dt        = 8e-5;
//__constant__ constexpr Real E         = 400;
//__constant__ constexpr int  dim       = 3;
//__constant__ constexpr int  steps     = 25;
//__constant__ constexpr int  neighbour = 27;
//__constant__ constexpr Real gravity   = 9.8;
//__constant__ constexpr int  bound     = 3;
//__constant__ constexpr Real p_rho     = 1.0;
//
//Vector* x_dev;
//Vector* v_dev;
//Matrix* C_dev;
//Real*   J_dev;
//Vector* grid_v_dev;
//Real*   grid_m_dev;
//
//__global__ void init_kernel(Real* J)
//{
//    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//    J[idx]   = 1;
//}
//
//__global__ void reset_kernel(Vector* grid_v, Real* grid_m)
//{
//    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//    grid_v[idx].setZero();
//    grid_m[idx] = 0;
//}
//
//template <class R, class A>
//__device__ R narrow_cast(const A& a)
//{
//    R r = R(a);
//    if(A(r) != a)
//        printf("warning: info loss in narrow_cast\n");
//    return r;
//}
//
//__device__ Vectori get_offset(size_t idx)
//{
//    Vectori offset;
//    for(auto i = dim - 1; i >= 0; i--)
//    {
//        offset[i] = narrow_cast<int, size_t>(idx % 3);
//        idx /= 3;
//    }
//    return offset;
//}
//
//__device__ Vectori get_indices(size_t idx, int n_grid)
//{
//    Vectori indices;
//    for(auto i = dim - 1; i >= 0; i--)
//    {
//        indices[i] = narrow_cast<int, size_t>(idx % n_grid);
//        idx /= n_grid;
//    }
//    return indices;
//}
//
//__global__ void particle_to_grid_kernel(Vector*     x,
//                                        Vector*     v,
//                                        Matrix*     C,
//                                        const Real* J,
//                                        Vector*     grid_v,
//                                        Real*       grid_m,
//                                        Real        dx,
//                                        Real        p_vol,
//                                        Real        p_mass,
//                                        int         n_grid)
//{
//    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//    // do not use the auto keyword with Eigen's expressions
//    Vector                Xp   = x[idx] / dx;
//    Vectori               base = (Xp.array() - 0.5f).cast<int>();
//    Vector                fx   = Xp - base.cast<Real>();
//    std::array<Vector, 3> w{0.5f * (1.5f - fx.array()).square(),
//                            0.75f - (fx.array() - 1.0f).square(),
//                            0.5f * (fx.array() - 0.5f).square()};
//    auto   stress = -dt * 4.f * E * p_vol * (J[idx] - 1.f) / (dx * dx);
//    Matrix affine = Matrix::Identity() * stress + p_mass * C[idx];
//    for(auto offset_idx = 0; offset_idx < neighbour; offset_idx++)
//    {
//        Vectori offset = get_offset(offset_idx);
//        Vector  dpos   = (offset.cast<Real>() - fx) * dx;
//        Real    weight = 1.0;
//        for(auto i = 0; i < dim; i++)
//        {
//            weight *= w[offset[i]][i];
//        }
//        // TODO: evaluate performance of atomic operations
//        Vector  grid_v_add      = weight * (p_mass * v[idx] + affine * dpos);
//        auto    grid_m_add      = weight * p_mass;
//        Vectori grid_idx_vector = base + offset;
//        auto    grid_idx        = 0;
//        for(auto i = 0; i < dim; i++)
//        {
//            grid_idx = grid_idx * n_grid + grid_idx_vector[i];
//        }
//        for(auto i = 0; i < dim; i++)
//        {
//            atomicAdd(&(grid_v[grid_idx][i]), grid_v_add[i]);
//        }
//        atomicAdd(&(grid_m[grid_idx]), grid_m_add);
//    }
//}
//
//__global__ void grid_update_kernel(Vector* grid_v, Real* grid_m, int n_grid)
//{
//    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
//    if(grid_m[idx] > 0)
//    {
//        grid_v[idx] /= grid_m[idx];
//    }
//    grid_v[idx][1] -= dt * gravity;
//    Vectori indices = get_indices(idx, n_grid);
//    for(auto i = 0; i < dim; i++)
//    {
//        if((indices[i] < bound && grid_v[idx][i] < 0)
//           || (indices[i] > n_grid - bound && grid_v[idx][i] > 0))
//        {
//            grid_v[idx][i] = 0;
//        }
//    }
//}
//
//__global__ void grid_to_particle_kernel(
//    Vector* x, Vector* v, Matrix* C, Real* J, Vector* grid_v, Real dx, int n_grid)
//{
//    auto                  idx  = blockIdx.x * blockDim.x + threadIdx.x;
//    Vector                Xp   = x[idx] / dx;
//    Vectori               base = (Xp.array() - 0.5f).cast<int>();
//    Vector                fx   = Xp - base.cast<Real>();
//    std::array<Vector, 3> w{0.5f * (1.5f - fx.array()).square(),
//                            0.75f - (fx.array() - 1.0f).square(),
//                            0.5f * (fx.array() - 0.5f).square()};
//    Vector                new_v = Vector::Zero();
//    Matrix                new_C = Matrix::Zero();
//    for(auto offset_idx = 0; offset_idx < neighbour; offset_idx++)
//    {
//        Vectori offset = get_offset(offset_idx);
//        Vector  dpos   = (offset.cast<Real>() - fx) * dx;
//        Real    weight = 1.0;
//        for(auto i = 0; i < dim; i++)
//        {
//            weight *= w[offset[i]][i];
//        }
//        Vectori grid_idx_vector = base + offset;
//        auto    grid_idx        = 0;
//        for(auto i = 0; i < dim; i++)
//        {
//            grid_idx = grid_idx * n_grid + grid_idx_vector[i];
//        }
//        new_v += weight * grid_v[grid_idx];
//        new_C += 4.0f * weight * grid_v[grid_idx] * dpos.transpose() / (dx * dx);
//    }
//    v[idx] = new_v;
//    x[idx] += dt * v[idx];
//    J[idx] *= Real(1.0) + dt * new_C.trace();
//    C[idx] = new_C;
//}
//
//class MPM
//{
//  public:
//    explicit MPM(int n_grid)
//        : n_grid(n_grid)
//    {
//        dim         = 3;
//        steps       = 25;
//        n_particles = utils::power(n_grid, dim) / utils::power(2, dim - 1);
//        neighbour   = power(3, dim);
//        dx          = 1.0 / n_grid;
//        p_rho       = 1.0;
//        p_vol       = power(dx * 0.5, 2);
//        p_mass      = p_vol * p_rho;
//        gravity     = 9.8;
//        bound       = 3;
//        E           = 400;
//    }
//
//    void init()
//    {
//        cudaFree(x_dev);
//        cudaFree(v_dev);
//        cudaFree(C_dev);
//        cudaFree(J_dev);
//        cudaFree(grid_v_dev);
//        cudaFree(grid_m_dev);
//
//        cudaMalloc(&x_dev, n_particles * sizeof(Vector));
//        cudaMalloc(&v_dev, n_particles * sizeof(Vector));
//        cudaMalloc(&C_dev, n_particles * sizeof(Matrix));
//        cudaMalloc(&J_dev, n_particles * sizeof(Real));
//        cudaMalloc(&grid_v_dev, power(n_grid, dim) * sizeof(Vector));
//        cudaMalloc(&grid_m_dev, power(n_grid, dim) * sizeof(Real));
//        cuda_check_error();
//
//        // initialize x on the host and copy to the device
//        auto x_host = std::make_unique<Vector[]>(n_particles);
//        for(auto i = 0; i < n_particles; i++)
//        {
//            for(auto j = 0; j < dim; j++)
//            {
//                x_host[i][j] = Real(rand_real());
//            }
//            x_host[i] = (x_host[i] * 0.4).array() + 0.15;
//        }
//        cudaMemcpy(x_dev, x_host.get(), n_particles * sizeof(Vector), cudaMemcpyHostToDevice);
//
//        cudaDeviceProp prop{};
//        cudaGetDeviceProperties(&prop, 0);
//        int block_dim{64};
//        threads_per_block = std::min(block_dim, prop.maxThreadsPerBlock);
//        auto block_num    = get_block_num(n_particles, threads_per_block);
//        init_kernel<<<block_num, threads_per_block>>>(J_dev);
//        cuda_check_error();
//    }
//
//    void advance()
//    {
//        auto T                  = steps;
//        auto particle_block_num = get_block_num(n_particles, threads_per_block);
//        auto grid_block_num = get_block_num(power(n_grid, dim), threads_per_block);
//        while(T--)
//        {
//            reset_kernel<<<grid_block_num, threads_per_block>>>(grid_v_dev, grid_m_dev);
//
//            particle_to_grid_kernel<<<particle_block_num, threads_per_block>>>(
//                x_dev, v_dev, C_dev, J_dev, grid_v_dev, grid_m_dev, dx, p_vol, p_mass, n_grid);
//
//            grid_update_kernel<<<grid_block_num, threads_per_block>>>(grid_v_dev, grid_m_dev, n_grid);
//
//            grid_to_particle_kernel<<<particle_block_num, threads_per_block>>>(
//                x_dev, v_dev, C_dev, J_dev, grid_v_dev, dx, n_grid);
//        }
//        cuda_check_error();
//    }
//
//    std::unique_ptr<Vector[]> to_numpy()
//    {
//        auto x_host = std::make_unique<Vector[]>(n_particles);
//        cudaMemcpy(x_host.get(), x_dev, n_particles * sizeof(Vector), cudaMemcpyDeviceToHost);
//
//        return x_host;
//    }
//
//    int get_n_particles() const { return n_particles; }
//
//  public:
//    int  dim         = 3;
//    int  n_grid      = 32;
//    int  steps       = 25;
//    int  n_particles = utils::power(n_grid, dim) / utils::power(2, dim - 1);
//    int  neighbour   = power(3, dim);
//    Real dx          = 1.0 / n_grid;
//    Real p_rho       = 1.0;
//    Real p_vol       = power(dx * 0.5, 2);
//    Real p_mass      = p_vol * p_rho;
//    Real gravity     = 9.8;
//    int  bound       = 3;
//    Real E           = 400;
//    int  threads_per_block;
//};
//
//int main(const int argc, const char** argv)
//{
//    int n_grid = 32;
//    if(argc > 1)
//    {
//        n_grid = atoi(argv[1]);
//    }
//    using namespace std::chrono_literals;
//
//    MPM* mpm = new MPM(n_grid);
//    // skip first run
//    mpm->init();
//    mpm->advance();
//    auto x = mpm->to_numpy();
//    int  num_frames{512};
//
//    auto start_time = std::chrono::high_resolution_clock::now();
//    for(auto runs = 0; runs < num_frames; runs++)
//    {
//        mpm->advance();
//        auto x = mpm->to_numpy();
//    }
//    auto end_time = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> diff = end_time - start_time;
//
//    float time_ms = diff.count() * 1000 / num_frames;
//
//    printf("{\"n_particles\":%d, \"time_ms\": %f}\n", mpm->get_n_particles(), time_ms);
//
//    return 0;
//}
//
//void mpm3d()
//{
//    example_desc(R"(This example we implement a simple mpm simulation.
//The source code is from:
//https://github.com/taichi-dev/taichi_benchmark/blob/main/suites/mpm/src/cuda/src/mpm3d.cu
//we rewrite it using muda compute graph.)");
//
//    int n_grid = 32;
//}
//
//TEST_CASE("mpm3d", "[pba]")
//{
//    mpm3d();
//}
