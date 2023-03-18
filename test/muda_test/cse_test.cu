#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>
#include <muda/composite/cse.h>
#include <muda/cub/device/device_scan.h>

using namespace muda;

void cse_test(host_vector<float>& res, host_vector<float>& ground_thruth)
{
    stream s;

    device_buffer_cse<float> dbcse(10, 2);
    dbcse.stream(s);
    device_buffer buf;
    //device_cse<float> dcse(10, 2);

    on(s)
        .next<launch>(1, 1)
        .apply(
            [begin = make_viewer(dbcse.begin), count = make_viewer(dbcse.count)] __device__() mutable
            {
                count(0) = 2;
                count(1) = 8;
            })
        .next<DeviceScan>()
        .ExclusiveSum(buf, dbcse.count.data(), dbcse.begin.data(), dbcse.dim_i())
        .next<parallel_for>(1, 32)
        .apply(dbcse.dim_i(),
               [cse = make_cse(dbcse)] __device__(const int i) mutable
               {
                   auto dimj = cse.dim_j(i);
                   for(int j = 0; j < dimj; ++j)
                       cse(i, j) = j;
               })
        .wait();
    dbcse.data.copy_to(res);
    for(size_t i = 0; i < 2; i++)
        ground_thruth.push_back(i);
    for(size_t i = 0; i < 8; i++)
        ground_thruth.push_back(i);
}

TEST_CASE("cse_test", "[cse]")
{
    host_vector<float> res, ground_thruth;
    cse_test(res, ground_thruth);
    REQUIRE(res == ground_thruth);
}
