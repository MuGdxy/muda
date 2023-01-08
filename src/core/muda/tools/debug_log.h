#pragma once
#include <cassert>
#include "fuzzy.h"
#include "../assert.h"
#include "../print.h"
#include "../muda_config.h"

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
                  __VA_ARGS__)

#define muda_debug_trap()                                                      \
    if constexpr(::muda::trapOnError)                                          \
        ::muda::trap();

#define muda_kernel_assert(res, fmt, ...)                                      \
    if(!(res))                                                                 \
    {                                                                          \
        muda_kernel_printf("<assert> " #res " failed." fmt "\n", __VA_ARGS__); \
        muda_debug_trap();                                                     \
    }

// check whether (res == true), if not, print the error info(never trap the device)
#define muda_kernel_check(res, fmt, ...)                                       \
    if(!(res))                                                                 \
    {                                                                          \
        muda_kernel_printf("<check> " #res " failed." fmt "\n", __VA_ARGS__);  \
    }

#define muda_kernel_error(fmt, ...)                                            \
    {                                                                          \
        muda_kernel_printf("<error> " fmt "\n", __VA_ARGS__);                  \
        muda_debug_trap();                                                     \
    }

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
                      __VA_ARGS__);                                            \
    }