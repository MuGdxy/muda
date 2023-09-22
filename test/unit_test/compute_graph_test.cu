#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/compute_graph/compute_graph.h>

using namespace muda;

void compute_graph_test()
{

    DeviceVector<float> a(10);

    auto view = make_viewer(a).name("view");

    ComputeGraph graph;

    auto var_view = graph.create_var<Dense1D<float>>("var_view");

    graph.add_node("set",
                   [&]
                   {
                       print("current phase: %d\n", (int)graph.current_graph_phase());
                       Launch(1, 1).apply(  //
                           [view = var_view->eval(), cview = var_view->ceval()] __device__() mutable
                           {
                               view(0) = cview(0) + 1;
                           });
                   });
    var_view->update(view);
    graph.launch();
    var_view->update(view.sub_view(1));
    graph.launch();

    HostVector<float> h(10);
    h = a;
    for(auto v : h)
        std::cout << v << " ";
    std::cout << std::endl;
}

TEST_CASE("compute_graph_test", "[default]")
{
    compute_graph_test();
}
