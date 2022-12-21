#include <catch2/catch.hpp>
#include <muda/muda.h>
#include "../example_common.h"
using namespace muda;

void graph_quick_start()
{
    example_desc("use muda to create graph.");
    device_var<int> value = 1;
    auto            graph = graph::create();

    // setup graph
    auto pA = parallel_for(1).asNodeParms(1, 
        [] __device__(int i) mutable { print("A\n"); });

    auto pB = parallel_for(1).asNodeParms(1, 
        [] __device__(int i) mutable { print("B\n"); });

    auto phB = host_call().asNodeParms(
        [&] __host__() { std::cout << "host" << std::endl; });

    auto pC = parallel_for(1).asNodeParms(1,
        [value = make_viewer(value)] __device__(int i) mutable
        {
            print("C, value=%d\n", value);
            value = 2;
        });

    auto pD = launch(1, 1).asNodeParms(
        [value = make_viewer(value)] __device__() mutable
        { print("D, value=%d\n", value); });

    auto kA = graph->addKernelNode(pA);
    auto kB = graph->addKernelNode(pB);
    auto hB = graph->addHostNode(phB, {kA, kB});
    auto kC = graph->addKernelNode(pC, {hB});
    auto kD = graph->addKernelNode(pD, {kC});

    auto instance = graph->instantiate();
    instance->launch();
}

TEST_CASE("graph_quick_start", "[quick_start]")
{
    graph_quick_start();
}