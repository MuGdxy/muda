#pragma once
#include <muda/muda_def.h>
#include <muda/tools/debug_break.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
namespace muda
{
MUDA_INLINE MUDA_GENERIC void trap() MUDA_NOEXCEPT
{
#ifdef __CUDA_ARCH__
    __trap();
#else
    throw std::exception("trap");
#endif
}

MUDA_INLINE MUDA_GENERIC void break_point() MUDA_NOEXCEPT
{
#ifdef __CUDA_ARCH__
    __brkpt();
#else
    debug_break();
#endif
}
}  // namespace muda