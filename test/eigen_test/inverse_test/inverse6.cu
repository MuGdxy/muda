#include "inverse_test.h"

TEST_CASE("inverse6", "[inverse_test]")
{
	inverse_test<float, 6>();
	inverse_test<double, 6>();
}
