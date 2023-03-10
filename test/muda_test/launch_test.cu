#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>

using namespace muda;

void launch_test(device_var<int>& res, device_var<int>& res2)
{
    stream s;
    launch(1, 1, 0, s)
        .apply(
            [res = make_viewer(res)] __device__() mutable
            {
                int a = 1;
                for(size_t i = 0; i < 1e15; i++)
                {
                    a /= 1;
                }
                res += 1;
            })
        .apply(
            [res = make_viewer(res)] __device__() mutable
            {
                if(res == 1)
                    res = 2;
            });

    memory(s).copy(data(res2), data(res), sizeof(int), cudaMemcpyDeviceToDevice);

    launch::wait_stream(s);
}

//TEST_CASE("launch_test", "[launch]")
//{
//    device_var<int> res = 0;
//    device_var<int> res2 = 0;
//    launch_test(res, res2);
//    int result = res;
//    int result2 = res2;
//    REQUIRE(result2 == 0);
//    REQUIRE(result == 2);
//}