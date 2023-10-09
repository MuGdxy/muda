#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/compute_graph/compute_graph_builder.h>
#include <muda/compute_graph/compute_graph.h>
#include <muda/compute_graph/compute_graph_var_manager.h>
#include <Eigen/Core>

using namespace muda;
using Vector3 = Eigen::Vector3f;

void compute_graph_simple()
{
    ComputeGraphVarManager manager;
    
    ComputeGraph graph{manager};


    DeviceVar<int> d;

    // define graph vars
    auto& N   = manager.create_var<size_t>("N");
    auto& x_0 = manager.create_var<Dense1D<Vector3>>("x_0");
    auto& x   = manager.create_var<Dense1D<Vector3>>("x");
    auto& y   = manager.create_var<Dense1D<Vector3>>("y");

    graph.create_node("cal_x_0") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x_0 = x_0.eval()] __device__(int i) mutable
                               { x_0(i) = Vector3::Ones(); });
    };

    graph.create_node("copy_to_x") << [&]
    {
        Memory().transfer(x.eval().data(), x_0.ceval().data(), N * sizeof(Vector3));
    };

    graph.create_node("copy_to_y") << [&]
    {
        Memory().transfer(y.eval().data(), x_0.ceval().data(), N * sizeof(Vector3));
    };

    graph.create_node("print_x_y") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x = x.ceval(), y = y.ceval(), N = N.eval()] __device__(int i) mutable
                               {
                                   if(N <= 10)
                                       printf("[%d] x = (%f,%f,%f) y = (%f,%f,%f) \n",
                                              i,
                                              x(i).x(),
                                              x(i).y(),
                                              x(i).z(),
                                              y(i).x(),
                                              y(i).y(),
                                              y(i).z());
                               });
    };

    // graph.graphviz(std::cout);

    Stream s;

    auto N_value    = 4;
    auto x_0_buffer = DeviceVector<Vector3>(N_value);
    auto x_buffer   = DeviceVector<Vector3>(N_value);
    auto y_buffer   = DeviceVector<Vector3>(N_value);

    N.update(N_value);
    x_0.update(x_0_buffer.viewer());
    x.update(x_buffer.viewer());
    y.update(y_buffer.viewer());


    graph.launch();
    graph.launch(true);
    Launch::wait_device();

    // update: change N
    auto f = [&](int new_N, int times)
    {
        N_value = new_N;
        x_0_buffer.resize(N_value);
        x_buffer.resize(N_value);
        y_buffer.resize(N_value);

        N.update(N_value);
        x_0.update(x_0_buffer.viewer());
        x.update(x_buffer.viewer());
        y.update(y_buffer.viewer());

        graph.update();

        auto t1 = profile_host(
            [&]
            {
                for(int i = 0; i < times; ++i)
                    graph.launch(s);
                Launch::wait_device();
            });
        auto t2 = profile_host(
            [&]
            {
                for(int i = 0; i < times; ++i)
                    graph.launch(s);
                Launch::wait_device();
            });

        std::cout << "N = " << N_value << std::endl;
        std::cout << "times = " << times << std::endl;
        std::cout << "graph launch time: " << t1 << "ms" << std::endl;
        std::cout << "single stream launch time: " << t2 << "ms" << std::endl
                  << std::endl;
    };

    //f(20, 1M);
}

void compute_graph_graphviz()
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
    ComputeGraphVarManager manager;
    ComputeGraph           graph{manager};

    auto& var_x_0 = manager.create_var("x_0", make_viewer(x_0));
    auto& var_h_x = manager.create_var("h_x", make_viewer(h_x));
    auto& var_x   = manager.create_var("x", make_viewer(x));
    auto& var_v   = manager.create_var("v", make_viewer(v));
    auto& var_toi = manager.create_var("toi", make_viewer(toi));
    auto& var_dt  = manager.create_var("dt", 0.1);
    auto& var_N   = manager.create_var("N", N);

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
        ParallelFor(256).apply(var_N,
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
        Memory().transfer(var_x_0.eval().data(), var_x.ceval().data(), var_N * sizeof(Vector3));
    };

    graph.create_node("download") << [&]
    {
        Memory().download(var_h_x.eval().data(), var_x.ceval().data(), var_N * sizeof(Vector3));
    };

    graph.graphviz(std::cout);
}

void compute_graph_multi_graph()
{
    ComputeGraphVarManager manager;
    // define graph vars
    ComputeGraph graph1{manager};

    ComputeGraph graph2{manager};


    DeviceVar<int> d;

    auto& N   = manager.create_var<size_t>("N");
    auto& x_0 = manager.create_var<Dense1D<Vector3>>("x_0");
    auto& x   = manager.create_var<Dense1D<Vector3>>("x");
    auto& y   = manager.create_var<Dense1D<Vector3>>("y");

    graph1.create_node("cal_x_0") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x_0 = x_0.eval()] __device__(int i) mutable
                               { x_0(i) = Vector3::Ones(); });
    };

    graph1.create_node("copy_to_x") << [&]
    {
        Memory().transfer(x.eval().data(), x_0.ceval().data(), N * sizeof(Vector3));
    };

    graph1.create_node("copy_to_y") << [&]
    {
        Memory().transfer(y.eval().data(), x_0.ceval().data(), N * sizeof(Vector3));
    };

    graph1.create_node("print_x_y") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x = x.ceval(), y = y.ceval(), N = N.eval()] __device__(int i) mutable
                               {
                               });
    };

    graph2.create_node("cal_x_0") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x_0 = x_0.eval()] __device__(int i) mutable
                               { x_0(i) = Vector3::Ones(); });
    };

    //std::cout << "graph 1=\n";
    //graph1.graphviz(std::cout);
    //std::cout << "graph 2=\n";
    //graph2.graphviz(std::cout);

    Stream s;

    auto N_value    = 4;
    auto x_0_buffer = DeviceVector<Vector3>(N_value);
    auto x_buffer   = DeviceVector<Vector3>(N_value);
    auto y_buffer   = DeviceVector<Vector3>(N_value);

    N.update(N_value);
    x_0.update(x_0_buffer.viewer());
    x.update(x_buffer.viewer());
    y.update(y_buffer.viewer());


    graph1.launch();
    graph2.launch();
    Launch::wait_device();
}

void compute_graph_update()
{
    ComputeGraphVarManager manager;
    // define graph vars
    ComputeGraph graph{manager};


    DeviceVar<int> d;

    auto& N   = manager.create_var<size_t>("N");
    auto& x_0 = manager.create_var<Dense1D<Vector3>>("x_0");
    auto& x   = manager.create_var<Dense1D<Vector3>>("x");
    auto& y   = manager.create_var<Dense1D<Vector3>>("y");

    graph.create_node("cal_x_0") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x_0 = x_0.eval()] __device__(int i) mutable
                               { x_0(i) = Vector3::Ones(); });
    };

    graph.create_node("copy_to_x") << [&]
    {
        Memory().transfer(x.eval().data(), x_0.ceval().data(), N * sizeof(Vector3));
    };

    graph.create_node("copy_to_y") << [&]
    {
        Memory().transfer(y.eval().data(), x_0.ceval().data(), N * sizeof(Vector3));
    };

    graph.create_node("print_x_y") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x = x.ceval(), y = y.ceval(), N = N.eval()] __device__(
                                   int i) mutable {});
    };

    // graph.graphviz(std::cout);

    Stream s;

    auto N_value    = 10;
    auto x_0_buffer = DeviceVector<Vector3>(N_value);
    auto x_buffer   = DeviceVector<Vector3>(N_value);
    auto y_buffer   = DeviceVector<Vector3>(N_value);

    N.update(N_value);
    x_0.update(x_0_buffer.viewer());
    x.update(x_buffer.viewer());
    y.update(y_buffer.viewer());


    graph.launch();

    N.update(N_value);
    x_0.update(x_0_buffer.viewer());
    x.update(x_buffer.viewer());
    y.update(y_buffer.viewer());

    graph.launch();
}


TEST_CASE("compute_graph_test_simple", "[compute_graph]")
{
    compute_graph_simple();
}

TEST_CASE("compute_graph_test_graphviz", "[compute_graph]")
{
	compute_graph_graphviz();
}

TEST_CASE("compute_graph_test_multi_graph", "[compute_graph]")
{
	compute_graph_multi_graph();
}

TEST_CASE("compute_graph_test_update", "[compute_graph]")
{
	compute_graph_update();
}
