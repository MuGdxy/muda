#pragma once
#include <muda/tools/extent.h>
#include <muda/mstl/span.h>
namespace muda::buffer::details
{
MUDA_INLINE bool masked_less(bool b, size_t l, size_t r) MUDA_NOEXCEPT
{
    return b ? l < r : true;
}
MUDA_INLINE bool masked_less(const std::array<bool, 2> mask,
                             const Offset2D&           lhs,
                             const Offset2D&           rhs) MUDA_NOEXCEPT
{
    return masked_less(mask[0], lhs.offset_in_width(), rhs.offset_in_width())
           && masked_less(mask[1], lhs.offset_in_height(), rhs.offset_in_height());
};

MUDA_INLINE bool masked_less(const std::array<bool, 3> mask,
                             const Offset3D&           lhs,
                             const Offset3D&           rhs) MUDA_NOEXCEPT
{
    return masked_less(mask[0], lhs.offset_in_width(), rhs.offset_in_width())
           && masked_less(mask[1], lhs.offset_in_height(), rhs.offset_in_height())
           && masked_less(mask[2], lhs.offset_in_depth(), rhs.offset_in_depth());
};
MUDA_INLINE bool masked_less(const std::array<bool, 2> mask,
                             const Extent2D&           lhs,
                             const Extent3D&           rhs) MUDA_NOEXCEPT
{
    return masked_less(mask[0], lhs.width(), rhs.width())
           && masked_less(mask[1], lhs.height(), rhs.height());
};

MUDA_INLINE bool masked_less(const std::array<bool, 3> mask,
                             const Extent3D&           lhs,
                             const Extent3D&           rhs) MUDA_NOEXCEPT
{
    return masked_less(mask[0], lhs.width(), rhs.width())
           && masked_less(mask[1], lhs.height(), rhs.height())
           && masked_less(mask[2], lhs.depth(), rhs.depth());
};
}  // namespace muda::buffer::details