#if MUDA_COMPUTE_GRAPH_ON
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <example_common.h>
#include <muda/syntax_sugar.h>
using namespace muda;

void hello_compute_graph()
{
    example_desc("This example we use ComputeGraph to say hello!");

    ComputeGraphVarManager manager;
    ComputeGraph           graph{manager, "graph_hello"};
    graph.$node("node_hello")  // syntax suger: create_node("hello") << [&]
    {
        Launch().apply(
            [] $()  // syntax suger: __device__ () mutable
            {
                // say hello
                print("hello compute graph!\n");
            });
    };
    graph.graphviz(std::cout);
    graph.launch();
    wait_device();
}

TEST_CASE("hello_compute_graph", "[compute_graph]")
{
    hello_compute_graph();
}

void hello_compute_graph_variable()
{
    example_desc("This example we introduce ComputeGraphVar");

    ComputeGraphVarManager manager;
    ComputeGraph           graph{manager, "graph_hello"};
    auto&                  var = manager.create_var<VarView<int>>("var");
    graph.$node("node_hello_1")
    {
        Launch().apply(
            [var = var.ceval().cviewer()] $()
            {
                // say hello
                print("hello compute graph! var = %d!\n", var);
            });
    };

    graph.$node("node_hello_2")
    {
        Launch().apply(
            [var = var.ceval().cviewer()] $()
            {
                // say hello
                print("hello compute graph! var = %d!\n", var);
            });
    };
    graph.graphviz(std::cout);

    // set device var value
    DeviceVar<int> var_value = 1;

    // update var value
    var = var_value;
    graph.launch();
    wait_device();
}

TEST_CASE("hello_compute_graph_variable", "[compute_graph]")
{
    hello_compute_graph_variable();
}
#endif