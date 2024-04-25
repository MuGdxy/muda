#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <example_common.h>
#include <muda/logger.h>
#include <Eigen/Core>
using namespace muda;

namespace muda
{
template <typename T, int M, int N>
__device__ LogProxy& operator<<(LogProxy& o, const Eigen::Matrix<T, M, N>& val)
{
    for(int i = 0; i < M; ++i, o << "\n")
        for(int j = 0; j < N; ++j)
            o << val(i, j) << " ";
    return o;
}
}  // namespace muda

void logger_overload()
{
    example_desc(R"(This example, we show how to overload muda::Logger output for your own type, 
e.g. for Eigen::Matrix you could write:
```c++
namespace muda
{
template <typename T, int M, int N>
__device__ LogProxy& operator<<(LogProxy& o, const Eigen::Matrix<T, M, N>& val)
{
    for(int i = 0; i < M; ++i, o << "\n")
        for(int j = 0; j < N; ++j)
            o << val(i, j) << " ";
    return o;
}
}  // namespace muda
```)");

    Logger logger;
    Launch(2, 1)
        .apply(
            [logger = logger.viewer()] __device__() mutable
            {
                Eigen::Matrix3f ones =
                    Eigen::Matrix3f::Ones() * static_cast<float>(blockIdx.x);
                logger << "blockIdx: (" << blockIdx << "):\n" << ones << "\n";
            })
        .wait();
    logger.retrieve(std::cout);
}

TEST_CASE("logger_overload", "[logger]")
{
    logger_overload();
}
