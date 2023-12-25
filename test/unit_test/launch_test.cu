#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/syntax_sugar.h>
using namespace muda;

struct MyTag
{
};
void launch_test()
{
    std::vector<int> gt;
    gt.resize(8 * 8 * 8, 1);
    DeviceBuffer<int> res(8 * 8 * 8);
    res.fill(0);

    Launch(cube(4))  // block dim = (4,4,4)
        .apply(
            cube(8),  // total count = (8,8,8)
            [res = make_dense_3d(res.data(), 8, 8, 8)] $(const int3 xyz)
            { res(xyz) = 1; },
            Tag<MyTag>{})
        .wait();

    std::vector<int> h_res;
    res.copy_to(h_res);

    REQUIRE(h_res == gt);

    gt.clear();
    gt.resize(4 * 4 * 4, 1);
    res.resize(4 * 4 * 4);
    res.fill(0);

    Launch(dim3{2, 2, 2}, dim3{2, 2, 2})
        .apply(
            [res = make_dense_3d(res.data(), make_int3(4, 4, 4))] $()
            {
                auto x       = threadIdx.x + blockIdx.x * blockDim.x;
                auto y       = threadIdx.y + blockIdx.y * blockDim.y;
                auto z       = threadIdx.z + blockIdx.z * blockDim.z;
                res(x, y, z) = 1;
            })
        .wait();

    res.copy_to(h_res);
    REQUIRE(h_res == gt);

    DeviceVar<int> block_dim;

    ParallelFor()
        .apply(100,
               [block_dim = block_dim.viewer()] $(int i)
               { block_dim = blockDim.x; })
        .wait();

    int h_block_dim = block_dim;
}

TEST_CASE("launch_test", "[launch]")
{
    launch_test();
}