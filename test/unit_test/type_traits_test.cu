#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>

using namespace muda;

template <typename F, typename... Args>
__global__ void is_invocable_kernel(int* result)
{
    *result = (int)std::is_invocable_v<F, Args...>;
}

template <typename F, typename... Args>
bool is_invocable_result()
{
    DeviceVar<int> result;
    Kernel{is_invocable_kernel<F, Args...>}(result.data());
    return result;
}

void type_traits_test()
{
    auto c1 = [] __device__() {};
    // warning: std::is_invocable_v can't detect lambda in host
    // the `std::is_invocable_v` is always `true`

    // REQUIRE(std::is_invocable_v<decltype(c1)> == true);
    // REQUIRE(std::is_invocable_v<decltype(c1), int, int> == false);

    // std::is_invocable_v can detect lambda in device.
    REQUIRE(is_invocable_result<decltype(c1)>() == true);
    REQUIRE(is_invocable_result<decltype(c1), int>() == false);

    auto c2 = [] __device__(const ParallelForDetails& f) {};
    auto c3 = [] __device__(int f) {};

    REQUIRE(is_invocable_result<decltype(c2), int>() == false);
    REQUIRE(is_invocable_result<decltype(c2), ParallelForDetails>() == true);
    REQUIRE(is_invocable_result<decltype(c3), int>() == true);
    REQUIRE(is_invocable_result<decltype(c3), ParallelForDetails>() == true);
}

TEST_CASE("type_traits_test", "[type_traits]")
{
    type_traits_test();
}
