#include <catch2/catch.hpp>
#include <muda/muda.h>

using namespace muda;

bool parallel_for_input_test(int count)
{
    bool pass = false;
    try
    {
        ParallelFor(32, 32).apply(count, [] __device__(int i) {}).wait();
        pass = true;
    }
    catch(std::logic_error)
    {
        pass = false;
    }
    return pass;
}

TEST_CASE("parallel_for_input", "[launch]")
{
    REQUIRE(parallel_for_input_test(1) == true);
    REQUIRE(parallel_for_input_test(-1) == false);
}