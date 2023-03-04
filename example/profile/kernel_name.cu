#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include "../example_common.h"
#include <numeric> // iota
using namespace muda;

void kernel_name()
{
    example_desc("use <profile> and <range_name> to help debugging and profiling.");
    host_vector<int> h_vec(100);
    device_vector<int> vec(100);
    device_var<int>    v(1);
    std::iota(h_vec.begin(), h_vec.end(), 0);
    vec = h_vec;

    {  //give a scope for RAII of profile and range_name

        // if you use Nsight System or Nsight Compute,
        // you can get the profile start point and see the kernel name
        profile p(__FUNCTION__);  // set a start point with function name for profiling.

        range_name r("kernel apply");  // give a name to this scope.
        parallel_for(32, 32)
            .apply(vec.size(),
                   [s = make_viewer(vec), v = make_viewer(v)] __device__(int i) mutable
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