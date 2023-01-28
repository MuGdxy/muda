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
        gm,       // add node to graph manager
        res       //res section
        {res::w,  // indicate that this kernel will write to res_var
         // any kernel that reads res_var latter will depend on this kernel
         res_var},
        [var = make_viewer(res_var)] __device__() mutable
        {
            var = 1;
            some_work(1e4);
            print("[A] set var=%d\n", var);
        },
        KernelATag{});

    parallel_for(res_vec.size())
        .addNode(
            gm,
            res{res_vec},  // indicate that this kernel will write to res_vec (default)
            res_vec.size(),  // parallel_for count
            [vec = make_viewer(res_vec)] __device__(int i) mutable
            {
                vec(i) = i;
                some_work(1e5);
                print("[B] set vec(%d)=%d\n", i, vec(i));
            },
            KernelBTag{});

    launch(1, 1).addNode(
        gm,
        res{res_var},  // indicate that this kernel will write to res_vec (default)
        [var = make_viewer(res_var)] __device__() mutable
        {
            some_work(1e4);
            auto next = 2;
            print("[A->C] set vev=%d -> %d\n", var, next);
            var = next;
        },
        KernelCTag{});

    launch(1, 1).addNode(
        gm,
        res{res::r,  // indicate that this kernel just read res_var (without modification)
            // any kernel that reads this resource won't depend on this kernel
            // the dependency will move upon the chain to the latest writer
            res_vec},
        [vec = make_viewer(res_vec)] __device__()
        {
            some_work(1e4);
            print("[B->D] vec={%d, %d}\n", vec(0), vec(1));
        },
        KernelDTag{});

    launch(1, 1).addNode(
        gm,       // add node to graph manager
        res       // res section
        {res::r,  // indicate that this kernel just read res_var and res_vec (without modification)
         // in this case, this kernel will depend on C, becuase C is the latest writer of res_var
         res_var,
         // in this case, this kernel won't depend on kernel D
         // because they just read the same resource without modification
         // but this kernel will depend on kernel B
         // because B is the latest writer of res_vec
         res_vec},
        [var = make_viewer(res_var), vec = make_viewer(res_vec)] __device__()
        {
            some_work(1e4);
            print("[(BC)->E] var=%d vec={%d, %d}\n", var, vec(0), vec(1));
        },
        KernelETag{});

    launch(1, 1).addNode(
        gm,  // add node to graph manager
        {},  // no res at all
        [] __device__() { print("[F] do nothing\n"); },
        KernelFTag{});

    auto instance = gm.instantiate();
    instance->launch();
    launch::wait_device();
}

TEST_CASE("graph_manager", "[graph]")
{
    graph_manager();
}
