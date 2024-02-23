#include "inverse_test.h"

TEST_CASE("inverse24", "[inverse_test]")
{
    inverse_test<float, 24>();
    inverse_test<double, 24>();
}
