#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/thread_only/mstring.h>
#include <muda/thread_only/vector.h>

using namespace muda;
using namespace muda::thread_only;
void string_test(string& res, string& ground_truth)
{
    ground_truth = "ab";

    device_vector<char> buf(ground_truth.size());
    launch(1, 1)
        .apply(
            [buf = make_viewer(buf)] __device__() mutable
            {
                string a  = "a";
                string b  = "b";
                string ab = a + b;
                buf(0)    = ab[0];
                buf(1)    = ab[1];
            })
        .wait();
    host_vector<char> hbuf = buf;
    for(size_t i = 0; i < hbuf.size(); i++)
        res.push_back(hbuf[i]);
}

TEST_CASE("string_test", "[thread_only]")
{
    string res, ground_truth;
    string_test(res, ground_truth);
    REQUIRE(res == ground_truth);
}
