#pragma once
#include "vector.h"
#include "thread_allocator.h"
#include "EASTL/priority_queue.h"

namespace muda::thread_only
{
template <typename T, typename Container = vector<T>, typename Compare = eastl::less<typename Container::value_type>>
using priority_queue = eastl::priority_queue<T, Container, Compare>;

template <typename T, typename Container = vector<T>>
using max_heap_queue =
    priority_queue<T, Container, eastl::less<typename Container::value_type>>;

template <typename T, typename Container = vector<T>>
using min_heap_queue =
    priority_queue<T, Container, eastl::greater<typename Container::value_type>>;
}  // namespace muda::thread_only