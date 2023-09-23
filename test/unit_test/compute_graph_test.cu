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
    DeviceVector<Vector3> x_0(N);
    DeviceVector<Vector3> x(N);
    DeviceVector<Vector3> v(N);
    DeviceVector<Vector3> dt(N);
    DeviceVar<float>      toi;
    HostVector<Vector3>   h_x(N);

    // define graph vars
    ComputeGraph graph;

    auto& var_x_0 = graph.create_var<Dense1D<Vector3>>("x_0", make_viewer(x_0));
    auto& var_h_x = graph.create_var<Dense1D<Vector3>>("h_x", make_viewer(h_x));
    auto& var_x   = graph.create_var<Dense1D<Vector3>>("x", make_viewer(x));
    auto& var_v   = graph.create_var<Dense1D<Vector3>>("v", make_viewer(v));
    auto& var_toi = graph.create_var<Dense<float>>("toi", make_viewer(toi));
    auto& var_dt  = graph.create_var<float>("dt", 0.1);
    auto& var_N   = graph.create_var<size_t>("N", N);

    // define graph nodes
    graph.create_node("cal_v") << [&]
    {
        ParallelFor(256)  //
            .apply(var_N,
                   [v = var_v.eval(), dt = var_dt.eval()] __device__(int i) mutable
                   {
                       // simple set
                       v(i) = Vector3::Ones() * dt * dt;
                   });
    };

    graph.create_node("cd") << [&]
    {
        ParallelFor(256)  //
            .apply(var_N,
                   [x   = var_x.ceval(),
                    v   = var_v.ceval(),
                    dt  = var_dt.eval(),
                    toi = var_toi.ceval()] __device__(int i) mutable
                   {
                       // collision detection
                   });
    };

    graph.create_node("cal_x") << [&]
    {
        ParallelFor(256).apply(var_N.eval(),
                               [x   = var_x.eval(),
                                v   = var_v.ceval(),
                                dt  = var_dt.eval(),
                                toi = var_toi.ceval()] __device__(int i) mutable
                               {
                                   // integrate
                                   x(i) += v(i) * toi * dt;
                               });
    };

    graph.create_node("transfer") << [&]
    {
        Memory().transfer(var_x_0.eval().data(), var_x.ceval().data(), N * sizeof(Vector3));
    };

    graph.create_node("download") << [&]
    {
        Memory().download(var_h_x.eval().data(), var_x.ceval().data(), N * sizeof(Vector3));
    };

    graph.graphviz(std::cout);
    graph.launch(true);
    graph.launch();
}

TEST_CASE("compute_graph_test", "[default]")
{
    compute_graph_test();
}
