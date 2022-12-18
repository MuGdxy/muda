#pragma once

#include <iostream>


#define example_desc(...) auto example_desc_ = example(__VA_ARGS__)


struct example
{
    example(const std::string& n, const std::string& description = "")
    {
        std::cout << "\033[1;32m";
        std::cout
            << "============================================================" << std::endl
            << "muda example: " << n << std::endl
            << "description: " << description << std::endl
            << "------------------------------------------------------------"
            << std::endl;
        std::cout << "\033[0m";
    }
    ~example()
    {
        std::cout << "\033[1;32m";
        std::cout << "============================================================"
                  << std::endl;
        std::cout << "\033[0m";
    }
};