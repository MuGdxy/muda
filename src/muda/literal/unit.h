#pragma once
#include <cinttypes>
namespace muda
{
constexpr size_t operator"" K(size_t value)
{
    return value * 1024;
}

constexpr size_t operator"" M(size_t value)
{
    return value * 1024 * 1024;
}

constexpr size_t operator"" G(size_t value)
{
    return value * 1024 * 1024 * 1024;
}

constexpr size_t operator"" T(size_t value)
{
    return value * 1024 * 1024 * 1024 * 1024;
}

constexpr size_t operator"" P(size_t value)
{
    return value * 1024 * 1024 * 1024 * 1024 * 1024;
}
}  // namespace muda