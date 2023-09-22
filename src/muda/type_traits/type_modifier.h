#pragma once
#include <type_traits>

namespace muda
{
template <typename T>
using raw_type_t = std::remove_all_extents_t<std::remove_reference_t<T>>;

template <typename T>
struct read_only_viewer
{
    using type = T;
};

template <typename T>
using read_only_viewer_t = typename read_only_viewer<T>::type;

template <typename T>
struct read_write_viewer
{
    using type = T;
};

template <typename T>
using read_write_viewer_t = typename read_write_viewer<T>::type;
}  // namespace muda