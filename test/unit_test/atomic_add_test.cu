#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <cuda.h>
#include <muda/container.h>
#include <muda/atomic.h>
#include <muda/syntax_sugar.h>

using namespace muda;

void atomic_add_test()
{
    DeviceVar<int> var    = 0;
    int            h_varA = 0;
    ParallelFor(32)
        .kernel_name("atomic_add_raw")
        .apply(256, [var = var.viewer()] $(int i) { atomicAdd(var.data(), 1); })
        .wait();
    h_varA = var;
    REQUIRE(h_varA == 256);

    int h_varB = 0;
    ParallelFor(32)
        .kernel_name("atomic_add")
        .apply(256, [var = var.viewer()] $(int i) { atomic_add(var.data(), 1); })
        .wait();
    h_varB = var;
    REQUIRE(h_varB == 512);
}

TEST_CASE("atomic_add_test", "[atomic_add]")
{
    atomic_add_test();
}
