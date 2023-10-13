#pragma once
#include <cinttypes>
namespace muda
{
constexpr size_t operator"" _K(size_t value)
{
    return value * 1024;
}

constexpr size_t operator"" _M(size_t value)
{
    return value * 1024 * 1024;
}

constexpr size_t operator"" _G(size_t value)
{
    return value * 1024 * 1024 * 1024;
}

constexpr size_t operator"" _T(size_t value)
{
    return value * 1024 * 1024 * 1024 * 1024;
}

constexpr size_t operator"" _P(size_t value)
{
    return value * 1024 * 1024 * 1024 * 1024 * 1024;
}
}  // namespace muda