#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#ifdef __CUDA_ARCH__
#define MUDA_GENERIC __host__ __device__
#else
#define MUDA_GENERIC
#endif
#define MUDA_THREAD_ONLY_AS_GENERIC

#ifdef MUDA_THREAD_ONLY_AS_GENERIC
#define MUDA_THREAD_ONLY MUDA_GENERIC
#else
#define MUDA_THREAD_ONLY __device__
#endif
