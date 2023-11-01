#include <catch2/catch.hpp>
#include <muda/muda.h>

using namespace muda;

__global__ void copy(int* dst, const int* src)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i];
}

void dynamic_parallelism(std::vector<int>& gt, std::vector<int>& res)
{
    gt.resize(16);
    res.resize(16);
    std::iota(gt.begin(), gt.end(), 0);

    DeviceBuffer<int> src = gt;

    DeviceBuffer<int> dst;
    dst.resize(gt.size());

    Launch()
        .kernel_name(__FUNCTION__)
        .apply(
            [src = src.cviewer(), dst = dst.viewer()] __device__() mutable {
                Kernel{1, 16, Stream::TailLaunch{}, copy}(dst.data(), src.data());
            });

    dst.copy_to(res);
}

TEST_CASE("dynamic_parallelism", "[dynamic_parallelism]")
{
    std::vector<int> gt, res;
    dynamic_parallelism(gt, res);
    REQUIRE(gt == res);
}
