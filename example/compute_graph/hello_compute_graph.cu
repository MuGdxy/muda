#include <catch2/catch.hpp>
#include <muda/muda.h>
#include "../example_common.h"
#include <muda/syntax_sugar.h>
using namespace muda;

void hello_compute_graph()
{
    example_desc("This example we use ComputeGraph to say hello!");

    ComputeGraphVarManager manager;
    ComputeGraph           graph{manager, "graph_hello"};
    graph.$node("node_hello") // syntax suger: create_node("hello") << [&]
    {
        Launch().apply([] $() // syntax suger: __device__ () mutable
        { 
            // say hello
            print("hello compute graph!"); 
        });
    };
    graph.graphviz(std::cout);
    graph.launch().wait();
}

TEST_CASE("hello_compute_graph", "[compute_graph]")
{
    hello_compute_graph();
}
