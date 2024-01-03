#pragma once
#include <muda/muda_def.h>
#include <muda/tools/debug_break.h>
#include <assert.h>
#include <exception>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <muda/exception.h>
#include <muda/check/check.h>

namespace muda
{
MUDA_INLINE MUDA_GENERIC void trap()
{
#ifdef __CUDA_ARCH__
    __trap();
#else
    std::abort();
#endif
}

MUDA_INLINE MUDA_GENERIC void brkpt() MUDA_NOEXCEPT
{
#ifdef __CUDA_ARCH__
    __brkpt();
#else
    debug_break();
#endif
}
}  // namespace muda