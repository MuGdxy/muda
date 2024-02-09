#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>
using namespace muda;

// there are some tag struct to label
// those kernel nodes in cuda graph.
// so that, when we use profile tools such as Nsight System,
// we can recognize them in the kernel name.
struct MyKernelB
{
};
struct MyKernelD
{
};

void graph_quick_start()
{
    example_desc(R"(use muda to create a graph manually:
            root
            /  \
            A  B
            \  / 
            host
             |
             C
             |
             D
)");

    auto graph = Graph::create();

    DeviceVar<int> value = 1;
    // setup graph
    auto pA = ParallelFor(1).as_node_parms(
        1, [] __device__(int i) mutable { print("kernel A\n"); });

    auto pB = ParallelFor(1).as_node_parms(
        1, [] __device__(int i) mutable { print("kernel B\n"); }, Tag<MyKernelB>{});

    auto pH = HostCall().as_node_parms([&] __host__()
                                       { std::cout << "host call" << std::endl; });

    auto pC = ParallelFor(1).as_node_parms(1,
                                           [value = value.viewer()] __device__(int i) mutable
                                           {
                                               print("kernel C, value=%d -> 2\n", value);
                                               value = 2;
                                           });

    auto pD =
        Launch(1, 1).as_node_parms([value = value.viewer()] __device__() mutable
                                   { print("kernel D, value=%d\n", value); },
                                   Tag<MyKernelD>{});

    auto kA = graph->add_kernel_node(pA);
    auto kB = graph->add_kernel_node(pB);
    auto h  = graph->add_host_node(pH, {kA, kB});
    auto kC = graph->add_kernel_node(pC, {h});
    auto kD = graph->add_kernel_node(pD, {kC});

    auto instance = graph->instantiate();
    instance->launch();
    wait_device();
}

TEST_CASE("graph_quick_start", "[quick_start]")
{
    graph_quick_start();
}