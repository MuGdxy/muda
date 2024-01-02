#if MUDA_COMPUTE_GRAPH_ON
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/cub/device/device_scan.h>
#include <muda/syntax_sugar.h>
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
    wait_device();

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
                wait_device();
            });
        auto t2 = profile_host(
            [&]
            {
                for(int i = 0; i < times; ++i)
                    graph.launch(s);
                wait_device();
            });

        std::cout << "N = " << N_value << std::endl;
        std::cout << "times = " << times << std::endl;
        std::cout << "graph launch time: " << t1 << "ms" << std::endl;
        std::cout << "single stream launch time: " << t2 << "ms" << std::endl
                  << std::endl;
    };

    //f(20, 1M);
}

TEST_CASE("compute_graph_simple", "[compute_graph]")
{
    compute_graph_simple();
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

    auto& var_x_0 = manager.create_var("x_0", x_0.view());
    auto& var_h_x = manager.create_var("h_x", h_x.data());
    auto& var_x   = manager.create_var("x", x.view());
    auto& var_v   = manager.create_var("v", v.view());
    auto& var_toi = manager.create_var("toi", toi.view());
    auto& var_dt  = manager.create_var("dt", 0.1);
    auto& var_N   = manager.create_var("N", N);

    // define graph nodes
    graph.create_node("cal_v") << [&]
    {
        ParallelFor(256)  //
            .apply(var_N,
                   [v = var_v.viewer(), dt = var_dt.eval()] __device__(int i) mutable
                   {
                       // simple set
                       v(i) = Vector3::Ones() * dt * dt;
                   });
    };

    graph.create_node("cd") << [&]
    {
        ParallelFor(256)  //
            .apply(var_N,
                   [x   = var_x.cviewer(),
                    v   = var_v.cviewer(),
                    dt  = var_dt.ceval(),
                    toi = var_toi.ceval()] __device__(int i) mutable
                   {
                       // collision detection
                   });
    };

    graph.create_node("cal_x") << [&]
    {
        ParallelFor(256).apply(var_N,
                               [x  = var_x.viewer(),
                                v  = var_v.cviewer(),
                                dt = var_dt.eval(),
                                toi = var_toi.cviewer()] __device__(int i) mutable
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
        Memory().download(var_h_x.eval(), var_x.ceval().data(), var_N * sizeof(Vector3));
    };

    graph.graphviz(std::cout);
}


TEST_CASE("compute_graph_graphviz", "[compute_graph]")
{
    compute_graph_graphviz();
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
                               [x = x.ceval(), y = y.ceval(), N = N.eval()] __device__(
                                   int i) mutable {});
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
    wait_device();
}

TEST_CASE("compute_graph_multi_graph", "[compute_graph]")
{
    compute_graph_multi_graph();
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
    wait_device();

    N.update(N_value);
    x_0.update(x_0_buffer.viewer());
    x.update(x_buffer.viewer());
    y.update(y_buffer.viewer());

    graph.launch();
    wait_device();
}

TEST_CASE("compute_graph_update", "[compute_graph]")
{
    compute_graph_update();
}

void compute_graph_buffer_view()
{
    ComputeGraphVarManager manager;

    ComputeGraph graph{manager};

    // define graph vars
    auto& N     = manager.create_var<size_t>("N");
    auto& x_0   = manager.create_var<BufferView<Vector3>>("x_0");
    auto& h_x_0 = manager.create_var<Vector3*>("h_x_0");
    auto& x     = manager.create_var<BufferView<Vector3>>("x");
    auto& y     = manager.create_var<BufferView<Vector3>>("y");

    graph.create_node("cal_x_0") << [&]
    {
        ParallelFor(256).kernel_name("cal_x_0").apply(
            N.eval(),
            [x_0 = x_0.eval().viewer()] __device__(int i) mutable
            { x_0(i) = Vector3::Ones(); });
    };

    graph.create_node("copy_to_h_x_0")
        << [&] { BufferLaunch().copy(h_x_0, x_0); };

    graph.create_node("copy_to_x") << [&] {  //
        BufferLaunch().copy(x, x_0);
    };

    graph.create_node("copy_to_y") << [&] { BufferLaunch().copy(y, x_0); };

    graph.create_node("print_x_y") << [&]
    {
        ParallelFor(256)
            .kernel_name("print_x_y")  //
            .apply(N.eval(),
                   [x = x.ceval().cviewer(), y = y.ceval().cviewer(), N = N.eval()] __device__(int i) mutable
                   {
                       if(N <= 10)
                           print("[%d] x = (%f,%f,%f) y = (%f,%f,%f) \n",
                                 i,
                                 x(i).x(),
                                 x(i).y(),
                                 x(i).z(),
                                 y(i).x(),
                                 y(i).y(),
                                 y(i).z());
                   });
    };

    graph.graphviz(std::cout);

    Stream stream;

    auto N_value      = 4;
    auto x_0_buffer   = DeviceBuffer<Vector3>(N_value);
    auto h_x_0_buffer = std::vector<Vector3>(N_value);
    auto x_buffer     = DeviceBuffer<Vector3>(N_value);
    auto y_buffer     = DeviceBuffer<Vector3>(N_value);

    N.update(N_value);
    x_0.update(x_0_buffer);
    h_x_0.update(h_x_0_buffer.data());
    x.update(x_buffer);
    y.update(y_buffer);

    // graph.launch(s);
    graph.launch(true, stream);

    stream.wait();
}

TEST_CASE("compute_graph_buffer_view", "[compute_graph]")
{
    compute_graph_buffer_view();
}

void compute_graph_one_closure_multi_graph_nodes()
{
    ComputeGraphVarManager manager;

    ComputeGraph graph{manager};

    // define graph vars
    auto& N     = manager.create_var<size_t>("N");
    auto& x_0   = manager.create_var<BufferView<Vector3>>("x_0");
    auto& h_x_0 = manager.create_var<Vector3*>("h_x_0");
    auto& x     = manager.create_var<BufferView<Vector3>>("x");
    auto& y     = manager.create_var<BufferView<Vector3>>("y");

    graph.create_node("cal_x_0_and_copy_to_h_x_0") << [&]
    {
        ParallelFor(256)
            .kernel_name("cal_x_0")  //
            .apply(N.eval(),
                   [x_0 = x_0.eval().viewer()] __device__(int i) mutable
                   { x_0(i) = Vector3::Ones(); });
        BufferLaunch().copy(h_x_0, x_0);
    };

    graph.create_node("copy_to_x") << [&] {  //
        BufferLaunch().copy(x, x_0);
    };

    graph.create_node("copy_to_y") << [&] { BufferLaunch().copy(y, x_0); };

    graph.create_node("print_x_y") << [&]
    {
        ParallelFor(256).apply(N.eval(),
                               [x = x.ceval().cviewer(),
                                y = y.ceval().cviewer(),
                                N = N.eval()] __device__(int i) mutable
                               {
                                   if(N <= 10)
                                       print("[%d] x = (%f,%f,%f) y = (%f,%f,%f) \n",
                                             i,
                                             x(i).x(),
                                             x(i).y(),
                                             x(i).z(),
                                             y(i).x(),
                                             y(i).y(),
                                             y(i).z());
                               });
    };

    ComputeGraphGraphvizOptions options;
    options.show_all_graph_nodes_in_a_closure = true;
    graph.graphviz(std::cout, options);

    Stream stream;

    auto N_value      = 4;
    auto x_0_buffer   = DeviceBuffer<Vector3>(N_value);
    auto h_x_0_buffer = std::vector<Vector3>(N_value);
    auto x_buffer     = DeviceBuffer<Vector3>(N_value);
    auto y_buffer     = DeviceBuffer<Vector3>(N_value);

    N.update(N_value);
    x_0.update(x_0_buffer);
    h_x_0.update(h_x_0_buffer.data());
    x.update(x_buffer);
    y.update(y_buffer);

    // graph.launch(s);
    graph.launch(stream);
    stream.wait();
    std::cout << "h_x_0_buffer = \n";
    for(auto& v : h_x_0_buffer)
        std::cout << v.transpose() << std::endl;
}

TEST_CASE("compute_graph_one_closure_multi_graph_nodes", "[compute_graph]")
{
    compute_graph_one_closure_multi_graph_nodes();
}

void compute_graph_capture()
{
    ComputeGraphVarManager manager;
    ComputeGraph           graph{manager};

    // define graph vars
    auto& N = manager.create_var<int>("N");
    auto& temp_storage = manager.create_var<BufferView<std::byte>>("temp_storage");
    auto& count  = manager.create_var<BufferView<int>>("count");
    auto& prefix = manager.create_var<BufferView<int>>("prefix");

    graph.$node("prefix_sum")
    {
        auto data = temp_storage.eval().data();
        auto size = temp_storage.eval().size();

        muda::DeviceScan().ExclusiveSum(
            data, size, count.ceval().data(), prefix.eval().data(), N.eval());
    };

    ComputeGraphGraphvizOptions options;
    options.show_all_graph_nodes_in_a_closure = true;
    graph.graphviz(std::cout, options);

    // prepare resources
    auto N_value      = 10;
    auto count_buffer = DeviceBuffer<int>(N_value);
    count_buffer.fill(1);
    auto   prefix_buffer = DeviceBuffer<int>(N_value);
    size_t temp_size;
    muda::DeviceScan().ExclusiveSum(
        nullptr, temp_size, count_buffer.data(), prefix_buffer.data(), N_value);
    auto temp_storage_buffer = DeviceBuffer<std::byte>(temp_size);

    // update graph vars
    N            = N_value;
    temp_storage = temp_storage_buffer;
    count        = count_buffer;
    prefix       = prefix_buffer;

    // graph.launch(s);
    graph.launch();

    std::vector<int> h_prefix(N_value);

    prefix_buffer.copy_to(h_prefix);

    std::vector<int> ground_truth(N_value);
    std::iota(ground_truth.begin(), ground_truth.end(), 0);

    REQUIRE(h_prefix == ground_truth);
}

TEST_CASE("compute_graph_capture", "[compute_graph]")
{
    compute_graph_capture();
}
#endif