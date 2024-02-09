#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/cuda/cooperative_groups.h>
#include <example_common.h>

using namespace muda;
namespace cg = cooperative_groups;

void show(DeviceVector<int>& values)
{
    HostVector<int> hvalues = values;
    std::cout << "values: " << std::endl;
    for(auto&& v : hvalues)
        std::cout << v << " ";
    std::cout << std::endl;
}

void grid_stride_vs_dynamic_grid()
{
    example_desc(
        "show the difference between grid-stride-loop and\n"
        "dynamic grid-stride-loop.\n"
        "grid stride loop uses fixed grid dim and block dim,\n"
        "while dynamic grid loop uses dynamic grid dim and block dim.");

    DeviceVector<int> values(16);

    //grid-stride loop
    ParallelFor(8, 8)
        .apply(values.size(),
               [values = values.viewer()] __device__(int i) mutable
               {
                   values(i) = i;
                   int gid   = cg::this_grid().thread_rank();
                   if(gid == 0)
                       muda::print("grid-stride loop: grid_size=%d, block_size=%d\n",
                                   gridDim.x,
                                   blockDim.x);
               })
        .wait();
    show(values);

    //dynamic-grid loop
    ParallelFor(8)
        .apply(values.size(),
               [values = values.viewer()] __device__(int i) mutable
               {
                   values(i) = i;
                   int gid   = cg::this_grid().thread_rank();
                   if(gid == 0)
                       muda::print("dynamic-grid loop: grid_size=%d, block_size=%d\n",
                                   gridDim.x,
                                   blockDim.x);
               })
        .wait();
    show(values);

    ParallelFor(2)
        .apply(values.size(),
               [values = values.viewer()] __device__(int i) mutable
               {
                   values(i) = i;
                   int gid   = cg::this_grid().thread_rank();
                   if(gid == 0)
                       muda::print("dynamic-grid loop: grid_size=%d, block_size=%d\n",
                                   gridDim.x,
                                   blockDim.x);
               })
        .wait();
    show(values);
}

TEST_CASE("grid_stride_vs_dynamic_grid", "[launch]")
{
    grid_stride_vs_dynamic_grid();
}