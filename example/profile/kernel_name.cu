#include <muda/muda.h>
#include <catch2/catch.hpp>
#include "../example_common.h"
using namespace muda;

void kernel_name()
{
    example_desc("use <profile> and <range_name> to help debugging and profiling.");
    universal_vector<int> h(100);
    universal_var<int>    v(1);
    for(size_t i = 0; i < h.size(); i++)
    {
        h[i] = i;
    }

    {  //give a scope for RAII of profile and range_name

        // if you use Nsight System or Nsight Compute,
        // you can get the profile start point and see the kernel name
        profile p(__FUNCTION__);  // set a start point with function name for profiling.

        range_name r("kernel apply");  // give a name to this scope.
        parallel_for(32, 32)
            .apply(h.size(),
                   [s = make_viewer(h), v = make_viewer(v)] __device__(int i) mutable
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