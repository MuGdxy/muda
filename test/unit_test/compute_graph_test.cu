#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/compute_graph/compute_graph.h>

using namespace muda;
using Vector3 = Eigen::Vector3f;
void compute_graph_test()
{
    // resources
    size_t                N = 10;
    DeviceVector<Vector3> x(N);
    DeviceVector<Vector3> v(N);
    DeviceVector<Vector3> dt(N);
    DeviceVar<float>      toi;

    // define graph vars
    ComputeGraph graph;
    auto& var_x   = graph.create_var<Dense1D<Vector3>>("x", make_viewer(x));
    auto& var_v   = graph.create_var<Dense1D<Vector3>>("v", make_viewer(v));
    auto& var_toi = graph.create_var<Dense<float>>("toi", make_viewer(toi));
    auto& var_dt  = graph.create_var<float>("dt", 0.1);
    auto& var_N   = graph.create_var<size_t>("N", N);

    // define graph nodes
    graph.create_node("cal_v") << [&]
    {
        ParallelFor(256).apply(var_N,
                               [v = var_v.eval(), dt = var_dt.eval()] __device__(int i) mutable
                               {
                                   // simple set
                                   v(i) = Vector3::Ones() * dt * dt;
                               });
    };

    graph.create_node("cd") << [&]
    {
        ParallelFor(256).apply(var_N,
                               [x = var_x.ceval()] __device__(int i) mutable
                               {
                                   // collision detection
                               });
    };

    graph.create_node("cal_x") << [&]
    {
        // print("current phase: %d\n", (int)ComputeGraphBuilder::current_phase());
        ParallelFor(256).apply(
            var_N.eval(),
            [x = var_x.eval(), v = var_v.ceval(), dt = var_dt.eval()] __device__(int i) mutable
            {
                // integrate
                x(i) += v(i) * dt;
            });
    };

    // launch graph
    graph.launch();
    graph.graphviz(std::cout);
    graph.launch(true);
    graph.launch();
}

TEST_CASE("compute_graph_test", "[default]")
{
    compute_graph_test();
}
