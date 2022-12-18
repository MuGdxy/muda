#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include "fuzzy.h"

#define muda_kernel_printf(fmt, ...)                                           \
    printf("(%d[%d],%d[%d],%d[%d])-(%d[%d],%d[%d],%d[%d]):" fmt,               \
           muda::block_idx().x,                                                \
           muda::grid_dim().x,                                                 \
           muda::block_idx().y,                                                \
           muda::grid_dim().y,                                                 \
           muda::block_idx().z,                                                \
           muda::grid_dim().z,                                                 \
           muda::thread_idx().x,                                               \
           muda::block_dim().x,                                                \
           muda::thread_idx().y,                                               \
           muda::block_dim().y,                                                \
           muda::thread_idx().z,                                               \
           muda::block_dim().z,                                                \
           __VA_ARGS__)
#define muda_kernel_assert(res)                                                \
    if(!(res))                                                                 \
    {                                                                          \
        muda_kernel_printf("assert(" #res ") fails\n");                        \
        assert(res);                                                           \
    }
namespace muda
{
template <typename... T>
inline __device__ void log_once(const char* fmt, T&&... args)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
        printf(fmt, args...);
}
}  // namespace muda