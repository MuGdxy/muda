#pragma once
#include <Eigen/Core>
template <typename T, int M, int N>
bool approx_equal(const Eigen::Matrix<T, M, N>& a, const Eigen::Matrix<T, M, N>& b)
{
    bool equal = a.isApprox(b);

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

        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                Eigen::Vector<T, 1> A{a(i, j)};
                Eigen::Vector<T, 1> B{b(i, j)};
                if(!A.isApprox(B))
                {
                    std::cout << "(" << i << ", " << j << ") => " << a(i, j)
                              << " != " << b(i, j) << std::endl;
                }
            }
        }

        std::cout << typeid(T).name()
                  << " eps: " << std::numeric_limits<T>::epsilon() << std::endl;
    }
    return equal;
};