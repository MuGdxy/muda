#include <catch2/catch.hpp>
#include <muda/muda.h>
using namespace muda;

void log_test()
{
    Logger logger;
    Launch(2, 2)
        .apply(
            [logger = logger.viewer()] __device__() mutable {  //
                logger << "threadIdx=(" << threadIdx << "): hello world\n";
            })
        .wait();
    logger.retrieve(std::cout);
}

TEST_CASE("log_test", "[log]")
{
    log_test();
}
