#if MUDA_COMPUTE_GRAPH_ON
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <example_common.h>
using namespace muda;

void compute_graph_event_record_and_wait()
{
    example_desc(R"(This example, we:
1) create 2 graphs
2) launch them on different streams, so they run concurrently
3) use event to synchronize them.)");


    ComputeGraphVarManager manager;

    ComputeGraph graph1{manager, "graph1"};
    ComputeGraph graph2{manager, "graph2"};

    auto& N     = manager.create_var<size_t>("N");
    auto& x     = manager.create_var<BufferView<int>>("x");
    auto& event = manager.create_var<cudaEvent_t>("event");

    graph1.create_node("set_x") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x = x.eval().viewer()] __device__(int i) mutable
                               {
                                   some_work();
                                   x(i) = 1;
                                   MUDA_KERNEL_PRINT("graph1 set x(%d) = %d", i, x(i));
                               });
    };

    graph1.create_node("event:record_event") << [&] { on().record(event, x); };

    graph1.create_node("do_some_work") << [&]
    {
        Launch().apply(
            // dummy read x, to make sure the kernel is launched after set_x
            [x = x.ceval().cviewer()] __device__() mutable
            {
                some_work();
                MUDA_KERNEL_PRINT("graph1 do some other work");
            });
    };

    graph2.create_node("event:wait_x") << [&] { on().wait(event, x); };

    graph2.create_node("read_x") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x = x.ceval().cviewer()] __device__(int i) mutable {
                                   MUDA_KERNEL_PRINT("graph2 read x(%d) = %d", i, x(i));
                               });
    };

    manager.graphviz(std::cout);

    Stream stream1;
    Stream stream2;


    auto N_value     = 4;
    auto x_value     = DeviceBuffer<int>(N_value);
    auto event_value = Event{};

    N     = N_value;
    x     = x_value;
    event = event_value;

    MUDA_KERNEL_PRINT("launch graph1 on stream1:");
    graph1.launch(stream1);
    MUDA_KERNEL_PRINT("launch graph2 on stream2:");
    graph2.launch(stream2);
    wait_device();
}

TEST_CASE("compute_graph_event_record_and_wait", "[compute_graph]")
{
    compute_graph_event_record_and_wait();
}
#endif