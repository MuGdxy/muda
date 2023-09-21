#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>

using namespace muda;

TEST_CASE("named_viewer_test", "[viewer]")
{
    auto v = Dense1D<float>(nullptr, 1);
    REQUIRE(v.name() == std::string("unnamed"));
    v.name("");
    REQUIRE(v.name() == std::string("unnamed"));
    v.name(nullptr);
    REQUIRE(v.name() == std::string("unnamed"));
    for(size_t i = 1; i < VIEWER_NAME_MAX; i++)
    {
        std::string s(i,'a');
        v.name(s.c_str());
        REQUIRE(v.name() == s);
    }
}
