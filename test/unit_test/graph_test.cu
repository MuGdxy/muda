#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <sstream>

using namespace muda;

struct kernelA
{
    kernelA(const muda::Dense1D<int>& s, const Dense<int>& var)
        : s(s)
        , var(var)
    {
    }

  public:
    Dense1D<int>    s;
    Dense<int>      var;
    __device__ void operator()(int i) { var = s.total_size(); }
};

// set_kernel_node_parms
void mem_realloc(int first, int last, int& outfirst, int& outlast)
{
    using namespace muda;
    // alloc device memory
    auto s   = DeviceVector<int>(first);
    auto var = DeviceVar<int>(0);

    // create kernel
    kernelA a(s.viewer(), var.viewer());

    // create graph
    auto graph = Graph::create();

    // setup graph node
    auto pA = ParallelFor(1).as_node_parms(1, a);
    auto kA = graph->add_kernel_node(pA);
    // create graph instance
    auto instance = graph->instantiate();

    // launch graph
    instance->launch();
    wait_device();
    outfirst = var;

    // realloc some device memory
    s.resize(last);
    // reset node parameters
    a.s = s.viewer();
    pA  = ParallelFor(1).as_node_parms(1, a);
    instance->set_kernel_node_parms(kA, pA);
    // luanch again
    instance->launch();
    wait_device();
    outlast = var;
}

TEST_CASE("set_graphExec_node_parms", "[graph]")
{
    for(int i = 10, j = 100; i < 100 && j < 1000; i += 10, j += 100)
    {
        std::stringstream ss;
        ss << "resize " << i << "->" << j;
        SECTION(ss.str())
        {
            int res_i, res_j;
            mem_realloc(i, j, res_i, res_j);
            REQUIRE(i == res_i);
            REQUIRE(j == res_j);
        }
    }
}

#ifdef MUDA_WITH_GRAPH_MEMORY_ALLOC_FREE
//// graph alloc
//void alloc_cpy_free(int half, host_vector<int>& host_data, host_vector<int>& ground_thruth)
//{
//    auto count = half * 2;
//    ground_thruth.resize(count);
//    host_data.resize(count);
//    for(int i = 0; i < ground_thruth.size(); i++)
//    {
//        ground_thruth[i] = i % half;
//    }
//
//    cudaStream_t stream;
//    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
//    auto hostDense = make_viewer(host_data);
//
//    auto  allocParm = memory::asAllocNodeParms<int>(count);
//    graph g;
//    auto [allocNode, ptr] = g.addMemAllocNode(allocParm);
//
//    auto Dense = make_dense(ptr, count);
//
//    auto writeKernelParm = ParallelFor(2, 8).as_node_parms(
//        count, [Dense = Dense] __device__(int i) mutable { Dense(i) = i; });
//    auto writeKernelNode = g.addKernelNode(writeKernelParm, {allocNode});
//    auto readKernelParm  = launch(1, 1).as_node_parms(
//        [Dense = Dense] __device__() mutable
//        {
//            for(int i = 0; i < Dense.total_size(); ++i)
//                ;
//        });
//    auto readKernelNode = g.addKernelNode(readKernelParm, {writeKernelNode});
//    auto cpyNode        = g.addMemcpyNode(
//        hostDense.data(), Dense.data(), half, cudaMemcpyDeviceToHost, {readKernelNode});
//    auto freeNode = g.addMemFreeNode(allocNode, {cpyNode});
//    auto instance = g.instantiate();
//    instance->launch(stream);
//    wait_stream(stream);
//    auto hostDenseHalf = Dense1D<int>(hostDense.data() + half, half);
//    instance->setMemcpyNodeParms(cpyNode, hostDenseHalf.data(), Dense.data(), half, cudaMemcpyDeviceToHost);
//    instance->launch(stream);
//    wait_stream(stream);
//
//    wait_device();
//}
//
//TEST_CASE("graph_memop_node", "[graph]")
//{
//    host_vector<int> v;
//    host_vector<int> ground_thruth;
//    for(int i = 50; i <= 1000; i += 50)
//    {
//        SECTION(std::to_string(i).c_str())
//        {
//            alloc_cpy_free(i, v, ground_thruth);
//            REQUIRE(v == ground_thruth);
//        }
//    }
//}
#endif

//void host_call_graph(int& ground_thruth, int& res)
//{
//    int v = 0;
//
//    auto hp = HostCall().as_node_parms([&v] __host__() mutable
//                                      { for(int i = 0; i < 5; ++i) v++; });
//
//    auto g = Graph::create();
//    g->add_host_node(hp);
//    auto instance = g->instantiate();
//    for(size_t i = 0; i < 10; i++)
//        instance->launch();
//    wait_device();
//
//    ground_thruth = 0;
//    for(size_t i = 0; i < 50; i++)
//        ground_thruth++;
//    res = v;
//}
//TEST_CASE("host_call_node", "[graph]")
//{
//    int ground_thruth, res;
//    host_call_graph(ground_thruth, res);
//    REQUIRE(ground_thruth == res);
//}