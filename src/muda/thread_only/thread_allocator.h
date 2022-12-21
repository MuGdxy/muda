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

//template <typename T>
//class global_allocator
//{
//    int mInitGid;
//
//  public:
//    using value_type = T;
//
//	MUDA_THREAD_ONLY global_allocator(const char* n)
//        : mInitGid(gtid())
//    {
//    }
//
//    MUDA_THREAD_ONLY global_allocator()
//        : mInitGid(gtid())
//    {
//    }
//
//    MUDA_THREAD_ONLY global_allocator(const global_allocator& x)
//    {
//        check_thread_only(x);
//        mInitGid = x.mInitGid;
//    }
//
//    MUDA_THREAD_ONLY global_allocator& operator=(const global_allocator& x)
//    {
//        check_thread_only(*this, x);
//        // there is no need to do any copy
//        return *this;
//    }
//
//    MUDA_THREAD_ONLY T* allocate(size_t n)
//    {
//        T* p;
//        checkCudaErrors(cudaMalloc(&p, n * sizeof(T)));
//        return p;
//    };
//	
//    T* allocate(size_t n, size_t alignment, size_t offset) {
//		
//    }
//	
//    MUDA_THREAD_ONLY void deallocate(T* p, size_t n)
//    {
//        checkCudaErrors(cudaFree(p));
//    }
//
//    MUDA_THREAD_ONLY char* name() { return "default allocator"; }
//
//    MUDA_THREAD_ONLY int init_gid() const { return mInitGid; }
//
//    friend MUDA_THREAD_ONLY bool operator==(const global_allocator& a,
//                                            const global_allocator& b)
//    {
//        return a.init_gid() == b.init_gid();
//    }
//    friend MUDA_THREAD_ONLY bool operator!=(const global_allocator& a,
//                                            const global_allocator& b)
//    {
//        return !operator==(a, b);
//    }
//
//  private:
//    MUDA_THREAD_ONLY static int bid()
//    {
//        return blockIdx.x + gridDim.x * blockIdx.y
//               + gridDim.x * gridDim.y * blockIdx.z;
//    }
//
//    MUDA_THREAD_ONLY static int tid()
//    {
//        return threadIdx.x + blockDim.x * threadIdx.y
//               + blockDim.x * blockDim.y * threadIdx.z;
//    }
//
//    MUDA_THREAD_ONLY static int gtid()
//    {
//        return bid() * blockDim.x * blockDim.y * blockDim.z + tid();
//    }
//
//    MUDA_THREAD_ONLY static void check_thread_only(const global_allocator& l)
//    {
//        if constexpr(debugThreadOnly)
//        {
//            auto gid = gtid();
//            if(l.mInitGid != gid)
//            {
//                muda_kernel_printf(
//                    "init_gid(%d) != current_gid(%d)\n"
//                    "thread only allocator should not be accessed by different threads\n",
//                    l.mInitGid,
//                    gid);
//                if constexpr(trapOnError)
//                    trap();
//            }
//        }
//    }
//
//    MUDA_THREAD_ONLY static void check_thread_only(const global_allocator& l, const global_allocator& r)
//    {
//        if constexpr(checkThreadOnlyAccess)
//        {
//            auto gid  = gtid();
//            auto prid = gid == l.mInitGid && gid == r.mInitGid;
//            if(!prid)
//            {
//                muda_kernel_printf(
//                    "allocator1.init_gid(%d) allocator2.init_gid(%d) current_gid(%d) not equal\n"
//                    "thread only allocator should not be accessed by different threads\n",
//                    l.mInitGid,
//                    r.mInitGid,
//                    gid);
//                if constexpr(trapOnError)
//                    trap();
//            }
//        }
//    }
//};
}  // namespace muda::thread_only
