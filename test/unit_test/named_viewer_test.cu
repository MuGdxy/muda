#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>

using namespace muda;

TEST_CASE("named_viewer_test", "[viewer]")
{
    auto v = Dense1D<float>(nullptr, 1);
    REQUIRE(v.name() == std::string("~"));
}
