#include <catch2/catch.hpp>
#include <muda/muda.h>

using namespace muda;

bool parallel_for_input_test(int begin, int end, int step)
{
    bool pass = false;
    try
    {
        parallel_for(32, 32).apply(begin, end, step, [] __device__(int i) {}).wait();
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
    REQUIRE(parallel_for_input_test(0, 99, 1) == true);
    REQUIRE(parallel_for_input_test(0, -10, -2) == true);

    REQUIRE(parallel_for_input_test(0, 0, 1) == false);
    REQUIRE(parallel_for_input_test(0, -1, 1) == false);
    REQUIRE(parallel_for_input_test(2, 1, 1) == false);
    REQUIRE(parallel_for_input_test(1, 100, 0) == false);
}