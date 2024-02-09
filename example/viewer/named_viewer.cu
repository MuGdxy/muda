#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>
using namespace muda;

void named_viewer()
{
    example_desc(R"(An example of naming a viewer.
Viewer name helps you navigate the error, if the viewer checker detects
the invalid access, you can navigate the source viewer.
ALERT: this example will ruin the cuda context and it's impossible 
to recover from the error!)");
    try
    {
        auto v = make_dense((int*)nullptr).name("my_viewer");
        // host access
        // print("v.name()=%s, v.kernel_name()=%s\n", v.name(), v.kernel_name());

        Launch(1, 1)
            .kernel_name("kernel_A")
            .apply(
                [v = make_dense_1d((int*)nullptr, 1).name("my_viewer")] __device__() mutable
                {
                    // device access
                    print("v.name()=%s, v.kernel_name()=%s\n", v.name(), v.kernel_name());
                    v(v.dim());
                })
            .wait();
    }
    catch(const std::exception& e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
    }
}

TEST_CASE("named_viewer", "[.]")
{
    named_viewer();
}
