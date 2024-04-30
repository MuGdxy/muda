#pragma once
#include <cstddef>
#include <muda/muda_def.h>
namespace muda
{
template <typename T>
class AsyncAllocator
{
  public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    T*                      allocate(size_type n);
    void                    deallocate(T* p, size_type n);
    MUDA_DEVICE static void construct(T* p, const T& val);
    MUDA_DEVICE static void destroy(T* p);
};
}  // namespace muda
#include "details/allocator.inl"