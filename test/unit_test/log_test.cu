#include <catch2/catch.hpp>
#include <muda/muda.h>
using namespace muda;

void log_test()
{
    Logger logger_;

    auto logger = logger_.viewer();
    Launch().apply([logger] __device__() mutable {  //
        logger << "threadIdx: " << threadIdx << "; blockIdx: " << blockIdx << "\n";
        int2 i2 = make_int2(1, 2);
        logger << "int2: " << i2 << "\n";
        float3 v3 = make_float3(1.0f, 2.0f, 3.0f);
        logger << "float3: " << v3 << "\n";

        LogProxy proxy{logger};
        int N[3] = {1, 2, 3};
        for(int i = 0; i < 3; ++i)
        {
            proxy << N[i] << " ";
        }
        proxy << "\n";
    });

    wait_device();
    logger_.retrieve(std::cout);
}

TEST_CASE("log_test", "[log]")
{
    log_test();
}
