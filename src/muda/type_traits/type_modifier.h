#pragma once
#include <type_traits>

namespace muda
{
template <typename T>
using raw_type_t = std::remove_all_extents_t<std::remove_reference_t<T>>;

template <typename T>
struct read_only_view
{
    using type = T;
};

template <typename T>
struct read_write_view
{
    using type = T;
};

template <typename T>
using read_only_view_t = typename read_only_view<T>::type;

template <typename T>
constexpr bool is_read_only_view_v = std::is_same_v<T, read_only_view_t<T>>;

template <typename T>
using read_write_view_t = typename read_write_view<T>::type;

template <typename T>
constexpr bool is_read_write_view_v = std::is_same_v<T, read_write_view_t<T>>;

template <typename T>
constexpr bool is_uniform_view_v = is_read_only_view_v<T> && is_read_write_view_v<T>;
}  // namespace muda