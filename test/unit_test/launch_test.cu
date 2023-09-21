#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>

using namespace muda;

void launch_test(DeviceVar<int>& res, DeviceVar<int>& res2)
{
    Stream s;
    on(s)
        .next<Launch>(1, 1)
        .apply([res = make_viewer(res)] __device__() mutable { res += 1; })
        .apply(
            [res = make_viewer(res)] __device__() mutable
            {
                if(res == 1)
                    res = 2;
            })
        .wait();
}

TEST_CASE("launch_test", "[launch]")
{
    DeviceVar<int> res  = 0;
    DeviceVar<int> res2 = 0;
    launch_test(res, res2);
    int result  = res;
    int result2 = res2;
    REQUIRE(result2 == 0);
    REQUIRE(result == 2);
}