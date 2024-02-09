#if MUDA_COMPUTE_GRAPH_ON
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <example_common.h>
using namespace muda;

void compute_graph_safe_var_update()
{
    example_desc(R"(This example, we create 2 trivial graphs, each only one node,
but we try to update the graph var `x` at different time. We show how to safely update
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
    N = N_value;
    x = x_value;

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
    graph2.launch(stream);
    wait_device();
}

TEST_CASE("compute_graph_safe_var_update", "[compute_graph]")
{
    compute_graph_safe_var_update();
}

void compute_graph_var_manager_update_var()
{
    example_desc(R"(This example, we create 3 trivial graphs, each only one node,
but we try to update the graph vars at different time. We show how to safely update
graph vars when you don't know whether some graphs are using your graph var or not. 
(This time we update var from var manager to achieve better performance.))");


    ComputeGraphVarManager manager;

    ComputeGraph graph1{manager};
    ComputeGraph graph2{manager};
    ComputeGraph graph3{manager};

    auto& N = manager.create_var<size_t>("N");
    auto& x = manager.create_var<BufferView<int>>("x");
    auto& y = manager.create_var<BufferView<int>>("y");

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

    graph2.create_node("set_y") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [y = y.eval().viewer()] __device__(int i) mutable
                               {
                                   y(i) = 2;
                                   // to make sure the graph1 is not finished so quickly
                                   some_work();
                               });
    };

    graph3.create_node("read_x_y") << [&]
    {
        ParallelFor(256).apply(
            N.eval(),
            [x = x.ceval().cviewer(), y = y.ceval().cviewer()] __device__(int i) mutable
            { print("x(%d) = %d, y(%d) = %d\n", i, x(i), i, y(i)); });
    };

    Stream stream;

    auto N_value = 4;
    auto x_value = DeviceBuffer<int>(N_value);
    auto y_value = DeviceBuffer<int>(N_value);

    // Frist update is safe, because no graph is using x
    N = N_value;
    x = x_value;
    y = y_value;

    graph1.launch(stream);
    graph2.launch(stream);

    // Second update is unsafe, because some graphs may be still using x and y

    auto is_using = manager.is_using(x, y);
    if(is_using)
    {
        std::cout << "graph1 or graph2 is still using x or y,\n"
                     "we call manager.sync(x, y) to wait for them to finish\n";
        manager.sync(x, y);  // direct call sync is also ok, you don't need to check is_using before.

        // manager.sync();  // sync all vars (convinient but not recommended, because we will wait for all vars)
    }
    // safely resize x and y
    N_value = 8;
    x_value.resize(N_value, 3);
    y_value.resize(N_value, 4);

    // update x_value, y_value and N_value to the var
    N = N_value;
    x = x_value;
    y = y_value;

    std::cout << " >>> Graph3 Results:\n";

    graph3.build();

    // this is a muda-style graph launch.
    // We keep graph launch and kernel launch consistent.
    on(stream)
        .next<GraphLaunch>()
        .launch(graph3)
        .next<Launch>()
        .apply([] __device__() { print("Graph3 is finished\n"); })
        .wait();
}

TEST_CASE("compute_graph_var_manager_update_var", "[compute_graph]")
{
    compute_graph_var_manager_update_var();
}
#endif