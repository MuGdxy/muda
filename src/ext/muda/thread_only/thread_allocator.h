#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "../muda_def.h"
#include "../muda_config.h"
#include "../tools/debug_log.h"
#include "../assert.h"
#include "../check/checkCudaErrors.h"
#include "EASTL/allocator.h"

MUDA_THREAD_ONLY inline void* operator new[](
    size_t size, const char* pName, int flags, unsigned debugFlags, const char* file, int line)
{
    return new char[size];
}

MUDA_THREAD_ONLY inline void* operator new[](size_t      size,
                                             size_t      alignment,
                                             size_t      alignmentOffset,
                                             const char* pName,
                                             int         flags,
                                             unsigned    debugFlags,
                                             const char* file,
                                             int         line)
{
    return new char[size];
}

namespace muda::thread_only
{
using thread_allocator = eastl::allocator;

/// <summary>
/// a thread stack allocator.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="Size"></typeparam>
template <typename T, int Size>
class thread_stack_allocator
{
    char buf_[Size * sizeof(T)];
    bool used_;

  public:
    MUDA_THREAD_ONLY thread_stack_allocator(const char*)
        : used_(false)
    {
    }

    MUDA_THREAD_ONLY static constexpr int size() { return Size; }

    MUDA_THREAD_ONLY bool is_using() const { return used_; }


    MUDA_THREAD_ONLY void* allocate(size_t n, int flags = 0)
    {
        if(n != 0)
        {
            muda_kernel_assert(!used_, "thread_static_allocator only allow allocating once, use container.reserve() for init!");
            muda_kernel_assert(n <= sizeof(buf_),
                               "allocation is too large (n=%d,buf=%d)",
                               int(n),
                               int(sizeof(buf_)));
            used_ = true;
            return buf_;
        }
    }
    MUDA_THREAD_ONLY void* allocate(size_t n, size_t alignment, size_t offset, int flags = 0)
    {
        return allocate(n, flags);
    }

    MUDA_THREAD_ONLY void deallocate(void* p, size_t n)
    {
        muda_kernel_assert(used_ && (char*)p == (char*)buf_, "deallocate fatal error");
        used_ = false;
    }
};

class external_buffer_allocator
{
    void* buf_;
    int   size_;
    bool  used_;
  public:
    MUDA_THREAD_ONLY external_buffer_allocator(void* ext_buf, int bytesize)
        : buf_(ext_buf)
        , size_(bytesize)
        , used_(false)
    {

    }

    MUDA_THREAD_ONLY int size() { return size_; }

    MUDA_THREAD_ONLY bool is_using() const { return used_; }

    MUDA_THREAD_ONLY void* allocate(size_t n, int flags = 0)
    {
        if(n != 0)
        {
            muda_kernel_assert(!used_, "external_buffer_allocator only allow allocating once, use container.reserve() for init!");
            muda_kernel_assert(n <= size_,
                               "allocation is too large (n=%d,buf=%d)",
                               int(n),
                               int(sizeof(size_)));
            used_ = true;
            return buf_;
        }
    }
    MUDA_THREAD_ONLY void* allocate(size_t n, size_t alignment, size_t offset, int flags = 0)
    {
        return allocate(n, flags);
    }

    MUDA_THREAD_ONLY void deallocate(void* p, size_t n)
    {
        muda_kernel_assert(used_ && (char*)p == (char*)buf_, "deallocate fatal error");
    }
};
}  // namespace muda::thread_only
