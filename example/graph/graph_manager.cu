#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include "../example_common.h"
using namespace muda;

struct KernelATag
{
};
struct KernelBTag
{
};
struct KernelCTag
{
};
struct KernelDTag
{
};
struct KernelETag
{
};
struct KernelFTag
{
};

void graph_manager()
{
    example_desc(
        R"(Use graphManager to setup a cuda graph,
which can automatically create dependencies from resources' read-write relation ship.
In this example, graphManager generates a graph like this: 
                          (R)
                         / | \
                        A  F  B
                        |    / \
                        C   /   D
                         \ /
                          E
in which, (R) is root, A B C D E F are kernels.
)");

    graphManager       gm;
    device_var<int>    res_var;
    device_vector<int> res_vec;
    res_vec.resize(2);

    launch(1, 1).addNode(
        gm,            // add node to graph manager
        res{res_var},  // has resource (default type = read-write): res_var
        [var = make_viewer(res_var)] __device__() mutable
        {
            var = 1;
            some_work(1e4);
            print("[A] set var=%d\n", var);
        },
        KernelATag{});

    parallel_for(res_vec.size())
        .addNode(
            gm,            // add node to graph manager
            res{res_vec},  // has resource (default type = read-write): res_vec
            res_vec.size(),  // parallel_for count
            [vec = make_viewer(res_vec)] __device__(int i) mutable
            {
                vec(i) = i;
                some_work(1e5);
                print("[B] set vec(%d)=%d\n", i, vec(i));
            },
            KernelBTag{});

    launch(1, 1).addNode(
        gm,            // add node to graph manager
        res{res_var},  // has resource (default type = read-write): res_var
        [var = make_viewer(res_var)] __device__() mutable
        {
            some_work(1e4);
            auto next = 2;
            print("[A->C] set vev=%d -> %d\n", var, next);
            var = next;
        },
        KernelCTag{});

    launch(1, 1).addNode(
        gm,  // add node to graph manager
        res  // res section
        {
            res::r,  // indicate the followers are read resources
            res_vec  // so any kernel that reads this resource won't depend directly on this kernel
        },
        [vec = make_viewer(res_vec)] __device__()
        {
            some_work(1e4);
            print("[B->D] vec={%d, %d}\n", vec(0), vec(1));
        },
        KernelDTag{});

    launch(1, 1).addNode(
        gm,  // add node to graph manager
        res  // res section
        {
            res::r,  // indicate the followers are read resources
            res_var,  // this kernel will depend on C, becuase C has written the res_var before
            res_vec  // and this kernel won't depend on kernel D although kernel D, but on kernel B
        },
        [var = make_viewer(res_var), vec = make_viewer(res_vec)] __device__()
        {
            some_work(1e4);
            print("[(BC)->E] var=%d vec={%d, %d}\n", var, vec(0), vec(1));
        },
        KernelETag{});

    launch(1, 1).addNode(
        gm,  // add node to graph manager
        {}, // no res at all
        [] __device__()
        {
            print("[F] do nothing\n");
        },
        KernelFTag{});

    auto instance = gm.instantiate();
    instance->launch();
    launch::wait_device();
}

TEST_CASE("graph_manager", "[graph]")
{
    graph_manager();
}
