#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>
#include <example_common.h>
using namespace muda;

void dense_viewer(HostVector<int>& ground_truth, HostVector<int>& res)
{
    example_desc(R"(an example for using dense viewer.)");

    cudaSetDevice(0);

    DeviceVar<int> scalar = 2;
    // thrust device_vector
    DeviceVector<int> vector(32, 1);
    // muda device_buffer
    DeviceVector<int> result;
    result.resize(32, 1);

    ParallelFor(32 /*blockDim*/)
        .apply(32 /*count*/,
               [scalar = scalar.viewer(),
                vector = vector.viewer(),
                buffer = result.viewer()] __device__(int i) mutable
               { buffer(i) = scalar * vector(i); })
        .wait();
    res = result;
    ground_truth.resize(32, 2);
}

TEST_CASE("dense_viewer", "[viewer]")
{
    HostVector<int> ground_truth, res;
    dense_viewer(ground_truth, res);
    REQUIRE(ground_truth == res);
}
