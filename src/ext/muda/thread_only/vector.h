#pragma once
#include "thread_allocator.h"
#include "EASTL/vector.h"
namespace muda::thread_only
{
template <typename T, typename Alloc = thread_allocator>
using vector = eastl::vector<T, Alloc>;
}