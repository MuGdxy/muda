#include <catch2/catch.hpp>
#include <muda/muda.h>
#include "../example_common.h"
using namespace muda;

void logger()
{
    example_desc(R"(This example, we show how to use `muda::Logger` and
show the advantages and disadvantages of `muda::Logger` w.r.t. the cuda `printf()`.

pro:
1. muda::Logger allow overloading, you can use it as a `std::ostream`,
   which means `logger << make_int2(0,0);` is allowed.
   see (<muda/logger/logger_function.h>) to learn how to overload your own type.
2. muda::Logger allow you to log in a dynamic loop while still keep the 
   output order against the parallel thread execution.

con:
1. if a cuda error is before your `muda::Logger::retrieve(std::cout)`, the result
   maybe unavailable.

flexibility:
1. to log something, you construct a `muda::Logger` yourself, and pass
   the `muda::Logger::viewer()` to your kernel.
2. to get the result, you call `muda::Logger::retrieve(ostream)`  
   by yourself, you could use a file or any `ostream` you like.
)");

    DeviceBuffer<int> dynamic_array(10);
    dynamic_array.fill(1);

    Logger logger;
    Launch(2, 1)
        .apply(
            [logger = logger.viewer()] __device__() mutable
            {
                //print hello world
                logger << "hello world! from block (" << blockIdx << ")\n";
            })
        .apply(
            [logger = logger.viewer(), dynamic_array = dynamic_array.viewer()] __device__() mutable
            {
                // type override
                int2 i2 = make_int2(1, 2);
                logger << "int2: " << i2 << "\n";
                float3 v3 = make_float3(1.0f, 2.0f, 3.0f);
                logger << "float3: " << v3 << "\n";

                // print a dynamic array and keep the ouput order
                LogProxy proxy{logger};
                proxy << "[thread=" << threadIdx.x << ", block=" << blockIdx.x << "]: ";
                for(int i = 0; i < dynamic_array.dim(); ++i)
                    proxy << dynamic_array(i) << " ";
                proxy << "(N=" << dynamic_array.dim() << ")\n";
            })
        .wait();
    logger.retrieve(std::cout);
}

TEST_CASE("logger", "[logger]")
{
    logger();
}
