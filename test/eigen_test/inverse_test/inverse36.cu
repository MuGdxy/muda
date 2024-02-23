#include "inverse_test.h"

TEST_CASE("inverse36", "[inverse_test]")
{
    inverse_test<float, 36>();
    inverse_test<double, 36>();
}
