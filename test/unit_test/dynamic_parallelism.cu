#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/syntax_sugar.h>

using namespace muda;

#ifndef __linux__
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
                Kernel{1, 16, Stream::FireAndForget{}, copy}(dst.data(), src.data());
            });

    dst.copy_to(res);
}

__global__ void simple_kernel()
{
    printf("simple_kernel\n");
}

TEST_CASE("dynamic_parallelism", "[dynamic_parallelism]")
{
    std::vector<int> gt, res;
    dynamic_parallelism(gt, res);
    REQUIRE(gt == res);
}
#endif


#if MUDA_COMPUTE_GRAPH_ON
void dynamic_parallelism_graph(std::vector<int>& gt, std::vector<int>& res)
{
    gt.resize(16);
    res.resize(16);
    std::iota(gt.begin(), gt.end(), 0);

    ComputeGraphVarManager manager;

    ComputeGraph      graph{manager, "graph", ComputeGraphFlag::DeviceLaunch};
    DeviceBuffer<int> src = gt;
    DeviceBuffer<int> dst(gt.size());

    auto& src_var = manager.create_var("src", src.view());
    auto& dst_var = manager.create_var("dst", dst.view());

    graph.$node("copy")
    {
        BufferLaunch().copy(dst_var, src_var);
    };
    graph.build();

    ComputeGraph launch_graph{manager, "launch_graph", ComputeGraphFlag::DeviceLaunch};
    auto& graph_var = manager.create_var("graph", graph.viewer());

    launch_graph.$node("launch")
    {
        Launch().apply([graph = graph_var.ceval()] $()
                       { graph.fire_and_forget(); });
    };

    manager.graphviz(std::cout);

    launch_graph.launch();
    wait_device();

    dst.copy_to(res);
}

#if MUDA_WITH_DEVICE_STREAM_MODEL
TEST_CASE("dynamic_parallelism_graph", "[dynamic_parallelism]")
{
    std::vector<int> gt, res;
    dynamic_parallelism_graph(gt, res);
    REQUIRE(gt == res);
}
#endif
#endif