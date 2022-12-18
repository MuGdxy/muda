#include <type_traits>

namespace muda
{
template <typename T>
using raw_type_t = std::remove_all_extents_t<std::remove_reference_t<T>>;
}