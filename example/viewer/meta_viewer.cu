#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include "../example_common.h"

using namespace muda;

void meta_viewer()
{
    example_desc("use meta-viewer <mapper> to create idxers.");
    device_vector<int> vec1(8, 2);
    device_vector<int> vec2(4, 1);
    device_var<int>    var;

    // make a mapper with size 4
    // mapper is a meta-viewer with out data pointer
    // which can be used to create certain viewers. eg. idxer
    auto vmap = make_mapper(4, vec1, vec2);

    parallel_for(16)
        .apply(0 /*begin*/,
               vec1.size() /*end*/,
               2 /*step*/,
               [     =, /*capture other variable by copying*/
                var  = make_viewer(var),
                vec1 = make_viewer(vec1)] __device__(int i) mutable
               { print("var=%d, i=%d, v=%d\n", var, i, vec1(i)); })
        .apply(4,/*count*/
               [     =, /*capture other variable by copying*/
                var  = make_viewer(var),
                vec2 = make_idxer(vmap, vec2)] __device__(int i) mutable
               { print("var=%d, i=%d, s=%d\n", var, i, vec2(i)); })
        .wait();
}

TEST_CASE("meta_viewer", "[quick_start]")
{
    meta_viewer();
}