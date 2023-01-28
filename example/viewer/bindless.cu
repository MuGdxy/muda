#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/atomic.h>
#include <muda/container.h>
#include "../example_common.h"
#include <cooperative_groups.h>
using namespace muda;

void bindless()
{
    example_desc(
        R"(This is an example for allocate and use memory in a bindless way:
in this example, we want to allocate an array, whose size is determined
by the kernel before, then pass it on to next kernel for latter usage,
and finally free it.)");

    device_var<dense1D<int>> bindless; // device_var to contain a dense1D viewer (we will allocate memory for it latter)
    device_var<int>          count = 0; // to contain the calculated allocation amount

    stream s;
    on(s).next<launch>(1, 8).apply(
        [  // count the array size to allocate
            count = make_viewer(count)] __device__() mutable
        {
            // do some collection operation to count the allocation amount.
            auto& data = *count;
            ::atomicAdd(&data, 1);
        });

    on(nullptr)
        .next<launch>(1, 1)  // launch a single thread to allocate memory
        .apply(
            [count = make_viewer(count), bindless = make_viewer(bindless)] __device__() mutable
            {
                auto array = new int[count];
                print("allocate an array of int with size %d\n", count);
                bindless = make_dense1D(array, count);
            });

    on(s) // do some work
        .next<launch>(1, 1)
        .apply(
            [bindless = make_viewer(bindless)] __device__() mutable
            {
                auto& viewer = *bindless;
                for(int i = 0; i < viewer.dim(); i++)
                    viewer(i) = i;
            })
        .apply(
            [bindless = make_viewer(bindless)] __device__() mutable
            {
                auto& viewer = *bindless;
                for(int i = 0; i < viewer.dim(); i++)
                    print("viewer(%d) = %d\n", i, viewer(i));
            });

    on(nullptr)
        .next<launch>(1, 1)  // launch a single thread to free memory
        .apply(
            [bindless = make_viewer(bindless)] __device__() mutable
            {
                delete[] bindless->data();
                print("delete array\n");
            })
        .wait();
}

TEST_CASE("bindless", "[viewer]")
{
    bindless();
}
