#pragma once
#include <cassert>
#include <thread>

#include "fuzzy.h"
#include "../assert.h"
#include "../print.h"
#include "../muda_config.h"
#include "../muda_def.h"

#ifdef __CUDA_ARCH__
#define muda_kernel_printf(fmt, ...)                                           \
    ::muda::print("(%d|%d,%d|%d,%d|%d)-(%d|%d,%d|%d,%d|%d):" fmt,              \
                  muda::block_idx().x,                                         \
                  muda::grid_dim().x,                                          \
                  muda::block_idx().y,                                         \
                  muda::grid_dim().y,                                          \
                  muda::block_idx().z,                                         \
                  muda::grid_dim().z,                                          \
                  muda::thread_idx().x,                                        \
                  muda::block_dim().x,                                         \
                  muda::thread_idx().y,                                        \
                  muda::block_dim().y,                                         \
                  muda::thread_idx().z,                                        \
                  muda::block_dim().z,                                         \
                  ##__VA_ARGS__)
#else
#define muda_kernel_printf(fmt, ...) ::muda::print("(host):" fmt, ##__VA_ARGS__)
#endif

//when muda::TRAP_ON_ERROR == true trap the device
#define muda_debug_trap()                                                      \
    if constexpr(::muda::TRAP_ON_ERROR)                                        \
        ::muda::trap();

// check whether (res == true), if not, print the error info (when muda::TRAP_ON_ERROR == true
// trap the device)
#define muda_kernel_assert(res, fmt, ...)                                      \
    if constexpr(!::muda::NO_CHECK)                                            \
    {                                                                          \
        if(!(res))                                                             \
        {                                                                      \
            muda_kernel_printf("%s(%d): %s <assert> " #res " failed." fmt,     \
                               __FILE__,                                       \
                               __LINE__,                                       \
                               MUDA_FUNCTION_SIG,                              \
                               ##__VA_ARGS__);                                 \
            muda_debug_trap();                                                 \
        }                                                                      \
    }

// check whether (res == true), if not, print the error info(never trap the device)
#define muda_kernel_check(res, fmt, ...)                                       \
    if constexpr(!::muda::NO_CHECK)                                            \
    {                                                                          \
        if(!(res))                                                             \
        {                                                                      \
            muda_kernel_printf("%s(%d): %s <check> " #res " failed." fmt,      \
                               __FILE__,                                       \
                               __LINE__,                                       \
                               MUDA_FUNCTION_SIG,                              \
                               ##__VA_ARGS__);                                 \
        }                                                                      \
    }

// print error info, and call muda_debug_trap()
#define muda_kernel_error(fmt, ...)                                            \
    {                                                                          \
        muda_kernel_printf("<error> " fmt, ##__VA_ARGS__);                     \
        muda_debug_trap();                                                     \
    }

// print warn info
#define muda_kernel_warn(fmt, ...)                                             \
    {                                                                          \
        muda_kernel_printf("<warn> " fmt, ##__VA_ARGS__);                      \
    }

// if certain debugOption == true, print the debug info
#define muda_kernel_debug_info(debugOption, fmt, ...)                          \
    if constexpr((debugOption))                                                \
    {                                                                          \
        ::muda::print("(%d|%d,%d|%d,%d|%d)-(%d|%d,%d|%d,%d|%d):" fmt           \
                      "(" #debugOption " = true)\n",                           \
                      ::muda::block_idx().x,                                   \
                      ::muda::grid_dim().x,                                    \
                      ::muda::block_idx().y,                                   \
                      ::muda::grid_dim().y,                                    \
                      ::muda::block_idx().z,                                   \
                      ::muda::grid_dim().z,                                    \
                      ::muda::thread_idx().x,                                  \
                      ::muda::block_dim().x,                                   \
                      ::muda::thread_idx().y,                                  \
                      ::muda::block_dim().y,                                   \
                      ::muda::thread_idx().z,                                  \
                      ::muda::block_dim().z,                                   \
                      ##__VA_ARGS__);                                          \
    }