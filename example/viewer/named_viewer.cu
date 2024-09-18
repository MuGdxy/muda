#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/buffer.h>
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
        DeviceBuffer<int> buffer(1);

        Launch(1, 1)
            .kernel_name("kernel_A")
            .file_line(__FILE__, __LINE__)
            .apply(
                [v = buffer.viewer().name("my_viewer")] __device__() mutable
                {
                    // device access
                    print("device: v.name()=%s, v.kernel_name()=%s, v.kernel_file()=%s, v.kernel_line()=%d\n",
                          v.name(),
                          v.kernel_name(),
                          v.kernel_file(),
                          v.kernel_line());
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
