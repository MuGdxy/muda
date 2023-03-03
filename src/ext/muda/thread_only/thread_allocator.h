#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <muda/muda_def.h>
#include <muda/muda_config.h>
#include <muda/tools/debug_log.h>
#include <muda/assert.h>
#include <muda/check/checkCudaErrors.h>
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
    char m_buf[Size * sizeof(T)];
    bool m_used;

  public:
    MUDA_THREAD_ONLY thread_stack_allocator(const char*)
        : m_used(false)
    {
    }

    MUDA_THREAD_ONLY static constexpr int size() { return Size; }

    MUDA_THREAD_ONLY bool is_using() const { return m_used; }


    MUDA_THREAD_ONLY void* allocate(size_t n, int flags = 0)
    {
        if(n != 0)
        {
            muda_kernel_assert(!m_used, "thread_static_allocator only allow allocating once, use container.reserve() for init!");
            muda_kernel_assert(n <= sizeof(m_buf),
                               "allocation is too large (n=%d,buf=%d)",
                               int(n),
                               int(sizeof(m_buf)));
            m_used = true;
            return m_buf;
        }
    }
    MUDA_THREAD_ONLY void* allocate(size_t n, size_t alignment, size_t offset, int flags = 0)
    {
        return allocate(n, flags);
    }

    MUDA_THREAD_ONLY void deallocate(void* p, size_t n)
    {
        muda_kernel_assert(m_used && (char*)p == (char*)m_buf, "deallocate fatal error");
        m_used = false;
    }
};

class external_buffer_allocator
{
    void* m_buf;
    int   m_size;
    bool  m_used;
  public:
    MUDA_THREAD_ONLY external_buffer_allocator(void* ext_buf, int bytesize)
        : m_buf(ext_buf)
        , m_size(bytesize)
        , m_used(false)
    {

    }

    MUDA_THREAD_ONLY int size() { return m_size; }

    MUDA_THREAD_ONLY bool is_using() const { return m_used; }

    MUDA_THREAD_ONLY void* allocate(size_t n, int flags = 0)
    {
        if(n != 0)
        {
            muda_kernel_assert(!m_used, "external_buffer_allocator only allow allocating once, use container.reserve() for init!");
            muda_kernel_assert(n <= m_size,
                               "allocation is too large (n=%d,buf=%d)",
                               int(n),
                               int(sizeof(m_size)));
            m_used = true;
            return m_buf;
        }
        return nullptr;
    }
    MUDA_THREAD_ONLY void* allocate(size_t n, size_t alignment, size_t offset, int flags = 0)
    {
        return allocate(n, flags);
    }

    MUDA_THREAD_ONLY void deallocate(void* p, size_t n)
    {
        muda_kernel_assert(m_used && (char*)p == (char*)m_buf, "deallocate fatal error");
    }
};
}  // namespace muda::thread_only
