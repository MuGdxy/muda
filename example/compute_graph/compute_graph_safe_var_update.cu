#include <catch2/catch.hpp>
#include <muda/muda.h>
#include "../example_common.h"
using namespace muda;

void compute_graph_safe_var_update()
{
    example_desc(R"(This example, we create 2 trivial graphs, each only one node,
but we try to update the graph var at different time. We show how to safely update
graph vars when you don't know whether some graphs are using your graph var.
1) graph1 sets x to 1s,
2) safe resize x
3) graph2 reads x and prints it)");


    ComputeGraphVarManager manager;

    ComputeGraph graph1{manager};
    ComputeGraph graph2{manager};

    auto& N = manager.create_var<size_t>("N");
    auto& x = manager.create_var<BufferView<int>>("x");

    graph1.create_node("set_x") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x = x.eval().viewer()] __device__(int i) mutable
                               {
                                   x(i) = 1;
                                   // to make sure the graph1 is not finished so quickly
                                   some_work();
                               });
    };

    graph2.create_node("read_x") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x = x.ceval().cviewer()] __device__(int i) mutable
                               { print("x(%d) = %d\n", i, x(i)); });
    };

    Stream stream;

    auto N_value = 4;
    auto x_value = DeviceBuffer<int>(N_value);

    // Frist update is safe, because no graph is using x
    N.update(N_value);
    x.update(x_value);

    graph1.launch(stream);

    // Second update is unsafe, because graph1 may still using x
    auto is_using = x.is_using();
    if(is_using)
    {
        std::cout << "graph1 is still using x, we call x.sync() to wait for it to finish\n";
        x.sync();  // direct call sync is also ok, you don't need to check is_using before.
    }
    std::cout << "graph1 is finished, now we can safely update x\n";
    N_value = 8;
    x_value.resize(N_value, 2);

    // update x_value and N_value
    N = N_value;
    x = x_value;

    std::cout << " >>> Graph2 Results:\n";
    graph2.launch(stream).wait();
}

TEST_CASE("compute_graph_safe_var_update", "[compute_graph]")
{
    compute_graph_safe_var_update();
}