
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/algorithm/prefix_sum.h>
using namespace muda;

void cub_test()
{
    device_vector<int> a(100);
    device_vector<int> b(100);
    device_vector<int> c(100);
    stream             s;
    on(s)
        .next(launch(1, 1))
        .apply([a = make_viewer(a)] __device__() mutable { a(0) = 1; })
        .next(DeviceScan(s))
        .ExclusiveSum(data(b), data(a), a.size())
        //.wait()
        .ExclusiveSum(data(c), data(b), a.size())
        //.wait()
        .next(launch(1, 1))
        .apply([c = make_viewer(c)] __device__() mutable
               { print("[99]=%d\n", c(99)); })
        .wait();
}

TEST_CASE("cub_test", "[cub]")
{
    cub_test();
}
