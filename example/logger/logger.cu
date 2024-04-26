#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/logger.h>
#include <example_common.h>
using namespace muda;

void logger_simple()
{
    example_desc(R"(A simple `muda::Logger` example)");

    Logger logger;
    Launch(2, 1)
        .apply(
            [logger = logger.viewer()] __device__() mutable
            {
                //print hello world
                logger << "hello world! from block (" << (uint3)blockIdx << ")\n";
            })
        .wait();
    logger.retrieve(std::cout);
}

TEST_CASE("logger_simple", "[logger]")
{
    logger_simple();
}

void log_proxy()
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
   by yourself, you could use a file or any `ostream` you like.)");

    std::vector<int> host_array(10);
    std::iota(host_array.begin(), host_array.end(), 0);
    DeviceBuffer<int> dynamic_array;
    dynamic_array = host_array;

    std::cout << "print a dynamic array using cuda `printf()` (out of order):\n";
    Launch(2, 1)
        .apply(
            [dynamic_array = dynamic_array.viewer()] __device__() mutable
            {
                printf("[thread=%d, block=%d]: ", threadIdx.x, blockIdx.x);
                for(int i = 0; i < dynamic_array.dim(); ++i)
                    printf("%d ", dynamic_array(i));
                printf("(N=%d)\n", dynamic_array.dim());
            })
        .wait();

    std::cout << "print a dynamic array and keep the output order using `muda::Logger`:\n";
    Logger logger;
    Launch(2, 1)
        .apply(
            [logger = logger.viewer(), dynamic_array = dynamic_array.viewer()] __device__() mutable
            {
                LogProxy proxy{logger};
                proxy << "[thread=" << threadIdx.x << ", block=" << blockIdx.x << "]: ";
                for(int i = 0; i < dynamic_array.dim(); ++i)
                    proxy << dynamic_array(i) << " ";
                proxy << "(N=" << dynamic_array.dim() << ")\n";
            })
        .wait();
    logger.retrieve(std::cout);
}

TEST_CASE("log_proxy", "[logger]")
{
    log_proxy();
}
