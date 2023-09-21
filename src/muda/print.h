#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace muda
{
template <typename InType, typename OutType = InType>
inline MUDA_GENERIC OutType print_convert(const InType& v)
{
    return v;
}

template <typename T>
inline MUDA_GENERIC const T& print_check(const T& t)
{
    static_assert(std::is_arithmetic_v<T> || std::is_pointer_v<T>
                      || std::is_same_v<T, std::nullptr_t>
                      || std::is_same_v<T, const char* const>
                      || std::is_same_v<T, const char*>,
                  "not supported type T in printf!");
    return t;
}

template <typename... Args>
inline MUDA_GENERIC void print(const char* const fmt, Args&&... arg)
{
    ::printf(fmt, print_check(print_convert(std::forward<Args>(arg)))...);
}
}  // namespace muda