#include "inverse_test.h"

TEST_CASE("inverse9", "[inverse_test]")
{
    inverse_test<float, 9>();
    inverse_test<double, 9>();
}
