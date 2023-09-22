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

    graph.add_node("node1",
                   [&]
                   {
                       print("current phase: %d\n", (int)graph.current_graph_phase());
                       const auto& view = var_view->ceval();
                   });

    graph.build();
}

TEST_CASE("compute_graph_test", "[default]")
{
    compute_graph_test();
}
