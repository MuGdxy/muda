#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include "../example_common.h"
using namespace muda;

void named_viewer()
{
    example_desc(R"(An example of naming a viewer.
Viewer name helps you navigate the error, if the viewer checker detects
the invalid access, you can navigate the source viewer.)");
    try
    {
        on(nullptr)
            .next<launch>()
            .apply(
                [v = make_viewer((int*)nullptr, 1)] __device__() mutable
                {
                    v.name("a_very_looooooooooong_name");
                    v.name("my_viewer");
                    print("v.name()=%s, v.dim()=%d.\n", v.name(), v.dim());
                    print("now, we are going to access v(v.dim()).\n");
                    v(v.dim());
                })
            .wait();
    }
    catch(std::exception e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
        cudaDeviceReset();
    }
}

TEST_CASE("named_viewer", "[.]")
{
    named_viewer();
}
