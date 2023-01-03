#pragma once
#include <type_traits>

namespace muda
{
template <typename _Ty, typename _TagTy, typename _VoidTy = void>
struct has_tag : std::false_type
{
};

template <typename _Ty, typename _TagTy>
struct has_tag<_Ty, _TagTy, std::void_t<typename _Ty::_TagTy>> : std::true_type
{
};

template <typename _Ty, typename _TagTy>
constexpr bool has_tag_v = has_tag<_Ty, _TagTy>::value;
}  // namespace muda