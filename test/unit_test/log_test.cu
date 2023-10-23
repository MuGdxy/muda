#include <catch2/catch.hpp>
#include <muda/muda.h>
using namespace muda;


MUDA_DEVICE LoggerViewer::Proxy& operator<<(LoggerViewer::Proxy& proxy, uint3 val)
{
    return proxy << val.x << "," << val.y << "," << val.z;
}

void log_test()
{
    Debug::init_logger();  // init global logger

    // global logger
    Launch(2, 2)
        .apply(
            [] __device__() mutable {  //
                cout << "threadIdx=(" << threadIdx << ") : hello world\n";
            })
        .wait();

    // user logger
    Logger logger{};
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
