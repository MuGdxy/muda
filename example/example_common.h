#pragma once

#include <iostream>


#define example_desc(...) example example_guard__(__FUNCTION__,__FILE__,__LINE__,__VA_ARGS__)


struct example
{
    example(const std::string& n, const std::string& file, int line,
        const std::string& description = "")
    {
        //get console width
        std::string name = "muda example: " + n;
        int         w    = 79;


        std::cout << std::string(w, '=') << std::endl;

        std::cout << "muda example: ";
        std::cout << "\033[1;32m";
        std::cout << n << std::endl;
        std::cout << "\033[0m";
        std::cout << "> " << file << "(" << line << ")" << std::endl;

        std::cout << "description: \n" << description << std::endl;
        
        std::cout << std::string(w, '-') << std::endl;
    }
    ~example()
    {
        std::cout << std::endl;
        //std::cout << "\033[1;32m";
        //std::cout << "==============================================================================="
        //          << std::endl;
        //std::cout << "\033[0m";
    }
};