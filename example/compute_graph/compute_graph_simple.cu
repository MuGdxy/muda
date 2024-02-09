#if MUDA_COMPUTE_GRAPH_ON
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>
using namespace muda;

void compute_graph_simple()
{
    example_desc(R"(This example create a compute graph with 4 nodes:
1) cal_x_0: set x_0 to 1.0s,
2) copy_to_x: copy x_0 to x,
3) copy_to_y: copy x_0 to y,
4) print_x_y: print x and y,
then we print the graphviz to the console, you can copy it to
https://dreampuf.github.io/GraphvizOnline/ to see the graph.
finally we launch the graph and get the result. 
)");

    ComputeGraphVarManager manager;

    ComputeGraph graph{manager};

    // define graph vars
    auto& N   = manager.create_var<size_t>("N");
    auto& x_0 = manager.create_var<BufferView<float>>("x_0");
    auto& x   = manager.create_var<BufferView<float>>("x");
    auto& y   = manager.create_var<BufferView<float>>("y");

    graph.create_node("cal_x_0") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x_0 = x_0.viewer()] __device__(int i) mutable
                               { x_0(i) = 1.0f; });
    };

    graph.create_node("copy_to_x") << [&] {  //
        BufferLaunch().copy(x.eval(), x_0.ceval());
    };

    graph.create_node("copy_to_y")
        << [&] { BufferLaunch().copy(y.eval(), x_0.ceval()); };

    graph.create_node("print_x_y") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x = x.cviewer(),
                                y = y.cviewer(),
                                N = N.eval()] __device__(int i) mutable
                               { print("[%d] x = %f y = %f \n", i, x(i), y(i)); });
    };

    graph.graphviz(std::cout);

    Stream stream;

    auto N_value    = 4;
    auto x_0_buffer = DeviceBuffer<float>(N_value);
    auto x_buffer   = DeviceBuffer<float>(N_value);
    auto y_buffer   = DeviceBuffer<float>(N_value);

    N   = N_value;
    x_0 = x_0_buffer;
    x   = x_buffer;
    y   = y_buffer;

    std::cout << "\n\n";
    std::cout << " >>> Results:\n";
    graph.launch(stream);
    wait_device();
}

TEST_CASE("compute_graph_simple", "[compute_graph]")
{
    compute_graph_simple();
}
#endif