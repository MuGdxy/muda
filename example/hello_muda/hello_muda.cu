#include <muda/muda.h>
#include <muda/buffer.h>
#include <catch2/catch.hpp>
#include <example_common.h>
using namespace muda;
void hello_muda()
{
    example_desc("say hello in muda");
    cudaSetDevice(1);
    Launch(1, 1).apply([] __device__() { print("hello muda!\n"); }).wait();
}

void quick_overview()
{
    example_desc("use parallel_for to say hello.");
    Stream s;
    on(s)
        .next(ParallelFor(2, 32))
        .apply(4,
               [] __device__(int i)
               { print("hello muda for %d/4 rounds\n", i + 1); })
        .wait();
}

TEST_CASE("muda_overview", "[quick_start]")
{
    quick_overview();
}

TEST_CASE("hello_muda", "[quick_start]")
{
    hello_muda();
}

