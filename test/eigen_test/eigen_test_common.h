#pragma once
#include <Eigen/Core>
template <typename T, int M, int N>
bool approx_equal(const Eigen::Matrix<T, M, N>& a, const Eigen::Matrix<T, M, N>& b)
{
    bool equal = true;
    if constexpr(std::is_same_v<T, float>)
        equal = (a - b).norm() < 1e-6;
    else
        equal = (a - b).norm() < 1e-12;
    if(!equal)
    {
        std::cout << "================================================================="
                  << std::endl;
        std::cout << a << std::endl;
        std::cout << "-----------------------------------------------------------------"
                  << std::endl;
        std::cout << b << std::endl;
        std::cout << "================================================================="
                  << std::endl;
    }
    return equal;
};