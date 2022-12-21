#pragma once
#include <cassert>
#include "fuzzy.h"
#include "../assert.h"
#include "../print.h"
#include "../muda_config.h"

#define muda_kernel_printf(fmt, ...)                                           \
    print("(%d[%d],%d[%d],%d[%d])-(%d[%d],%d[%d],%d[%d]):" fmt,                \
          muda::block_idx().x,                                                 \
          muda::grid_dim().x,                                                  \
          muda::block_idx().y,                                                 \
          muda::grid_dim().y,                                                  \
          muda::block_idx().z,                                                 \
          muda::grid_dim().z,                                                  \
          muda::thread_idx().x,                                                \
          muda::block_dim().x,                                                 \
          muda::thread_idx().y,                                                \
          muda::block_dim().y,                                                 \
          muda::thread_idx().z,                                                \
          muda::block_dim().z,                                                 \
          __VA_ARGS__)

#define muda_kernel_assert(res, fmt, ...)                                      \
    if(!(res))                                                                 \
    {                                                                          \
        muda_kernel_printf("assert failed " #res fmt, __VA_ARGS__);            \
        if constexpr(::muda::trapOnError)                                      \
            ::muda::trap();                                                    \
    }

#define muda_kernel_check(res, fmt, ...)                                       \
    if(!(res))                                                                 \
    {                                                                          \
        muda_kernel_printf("check failed " #res fmt, __VA_ARGS__);             \
    }

#define muda_kernel_error(fmt, ...)                                            \
    {                                                                          \
        muda_kernel_printf(fmt, __VA_ARGS__);                                  \
        if constexpr(::muda::trapOnError)                                      \
            ::muda::trap();                                                    \
    }