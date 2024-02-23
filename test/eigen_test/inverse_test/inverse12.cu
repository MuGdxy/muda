#include "inverse_test.h"

TEST_CASE("inverse12", "[inverse_test]")
{
    inverse_test<float, 12>();
    inverse_test<double, 12>();
}
