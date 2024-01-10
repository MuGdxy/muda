#pragma once
#include <muda/type_traits/type_modifier.h>

namespace muda
{
template <bool IsConst>
class ViewBase
{
  public:
    constexpr static bool IsConst    = IsConst;
    constexpr static bool IsNonConst = !IsConst;
    template <typename T>
    using auto_const_t = std::conditional_t<IsConst, const T, T>;
    template <typename T>
    using non_const_enable_t = std::enable_if_t<IsNonConst, T>;
};
}  // namespace muda