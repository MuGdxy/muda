#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include "../example_common.h"
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
    example_desc(R"(use muda to create a graph:
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
	
    auto graph = graph::create();

    device_var<int> value = 1;
    // setup graph
    auto pA = parallel_for(1).asNodeParms(
        1, [] __device__(int i) mutable { print("kernel A\n"); });

    auto pB = parallel_for(1).asNodeParms(
        1, [] __device__(int i) mutable { print("kernel B\n"); }, MyKernelB{});

    auto pH = host_call().asNodeParms([&] __host__()
                                      { std::cout << "host call" << std::endl; });

    auto pC =
        parallel_for(1).asNodeParms(1,
                                    [value = make_viewer(value)] __device__(int i) mutable
                                    {
                                        print("kernel C, value=%d -> 2\n", value);
                                        value = 2;
                                    });

    auto pD = launch(1, 1).asNodeParms([value = make_viewer(value)] __device__() mutable
                                       { print("kernel D, value=%d\n", value); },
                                       MyKernelD{});

    auto kA = graph->addKernelNode(pA);
    auto kB = graph->addKernelNode(pB);
    auto h  = graph->addHostNode(pH, {kA, kB});
    auto kC = graph->addKernelNode(pC, {h});
    auto kD = graph->addKernelNode(pD, {kC});

    auto instance = graph->instantiate();
    instance->launch();
}

TEST_CASE("graph_quick_start", "[quick_start]")
{
    graph_quick_start();
}