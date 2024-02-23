#include "inverse_test.h"

TEST_CASE("inverse_small", "[inverse_test]")
{
    inverse_test<float, 2, eigen::AnalyticalInverse>();
    inverse_test<float, 3, eigen::AnalyticalInverse>();
    inverse_test<float, 4, eigen::AnalyticalInverse>();

    inverse_test<double, 2, eigen::AnalyticalInverse>();
    inverse_test<double, 3, eigen::AnalyticalInverse>();
    inverse_test<double, 4, eigen::AnalyticalInverse>();

    inverse_test<float, 2, eigen::GaussEliminationInverse>();
    inverse_test<float, 3, eigen::GaussEliminationInverse>();
    inverse_test<float, 4, eigen::GaussEliminationInverse>();

    inverse_test<double, 2, eigen::GaussEliminationInverse>();
    inverse_test<double, 3, eigen::GaussEliminationInverse>();
    inverse_test<double, 4, eigen::GaussEliminationInverse>();
}