#pragma once

#include <iostream>
#include <fstream>

#define example_desc(...)                                                      \
    example example_guard__(__FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)

constexpr const char* white       = "\033[1,37m";
constexpr const char* yellow      = "\033[1;33m";
constexpr const char* green       = "\033[1;32m";
constexpr const char* blue        = "\033[1;34m";
constexpr const char* cyan        = "\033[1;36m";
constexpr const char* red         = "\033[1;31m";
constexpr const char* magenta     = "\033[1;35m";
constexpr const char* black       = "\033[1;30m";
constexpr const char* darkwhite   = "\033[0;37m";
constexpr const char* darkyellow  = "\033[0;33m";
constexpr const char* darkgreen   = "\033[0;32m";
constexpr const char* darkblue    = "\033[0;34m";
constexpr const char* darkcyan    = "\033[0;36m";
constexpr const char* darkred     = "\033[0;31m";
constexpr const char* darkmagenta = "\033[0;35m";
constexpr const char* darkblack   = "\033[0;30m";
constexpr const char* off         = "\033[0;0m";

struct color
{
    color(const char* c) { std::cout << c; }
    ~color() { std::cout << off; }
};

struct example
{
    example(const std::string& n,
            const std::string& file,
            int                line,
            const std::string& description = "",
            bool               show_code   = false)
    {
        std::cout << std::dec;
        //get console width
        std::string name = "muda example: " + n;
        int         w    = 79;


        std::cout << std::string(w, '=') << std::endl;

        std::cout << "muda example: ";

        {
            color c(blue);
            std::cout << n << std::endl;
        }

        {
            color c(darkyellow);
            std::cout << "> " << file << "(" << line << ")" << std::endl;
        }

        std::cout << "description: \n" << description << std::endl;


        if(show_code)
        {
            std::cout << std::string(w, '-') << std::endl;
            std::ifstream ifs(file);
            if(ifs.is_open())
                std::cout << ifs.rdbuf() << std::endl;
        }
        std::cout << std::string(w, '-') << std::endl;

        {
            color c(green);
            std::cout << "output:" << std::endl;
        }
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

// a dummy function, just let the kernel wait some clock cycles
// to make the kernel execution time long enough.
inline __device__ void some_work(size_t clock_cycles = 1e9)
{
    clock_t start = clock();
    clock_t now;
    while(true)
    {
        now            = clock();
        clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if(cycles >= clock_cycles)
            break;
    }
}

inline void make_progress_bar(float progress, int bar = 77)
{
    int         w = std::round(progress * bar);
    std::string done(w, '>');
    std::string undone(bar - w, '=');
    std::cout << "[" << done << undone << "]\r";
}