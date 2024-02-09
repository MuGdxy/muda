#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>
#include <numeric> // iota
using namespace muda;

void kernel_name()
{
    example_desc("use <profile> and <range_name> to help debugging and profiling.");
    HostVector<int> h_vec(100);
    DeviceVector<int> vec(100);
    DeviceVar<int>    v(1);
    std::iota(h_vec.begin(), h_vec.end(), 0);
    vec = h_vec;

    {  //give a scope for RAII of profile and range_name

        // if you use Nsight System or Nsight Compute,
        // you can get the profile start point and see the kernel name
        Profile p(__FUNCTION__);  // set a start point with function name for profiling.

        RangeName r("kernel apply");  // give a name to this scope.
        ParallelFor(32, 32)
            .kernel_name("my_parallel_for")  // set a kernel name for debugging in kernel
            .apply(vec.size(),
                   [s = vec.viewer(), v = v.viewer()] __device__(int i) mutable
                   {
                       __shared__ int b[64];
                       s(i)           = v;
                       b[threadIdx.x] = v;
                   })
            .wait();
    }
}

TEST_CASE("kernel_name", "[profile]")
{
    kernel_name();
}