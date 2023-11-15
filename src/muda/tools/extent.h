#pragma once
#include <cuda.h>
#include <muda/muda_def.h>
#include <cinttypes>

namespace muda
{
class Extent2D
{
    size_t m_extent[2];

  public:
    MUDA_GENERIC Extent2D() MUDA_NOEXCEPT : m_extent{~0, ~0} {}

    MUDA_GENERIC Extent2D(size_t height, size_t width) MUDA_NOEXCEPT
        : m_extent{height, width}
    {
    }

    MUDA_GENERIC size_t height() const MUDA_NOEXCEPT { return m_extent[0]; }
    MUDA_GENERIC size_t width() const MUDA_NOEXCEPT { return m_extent[1]; }

    template <typename T>
    MUDA_GENERIC cudaExtent cuda_extent() const MUDA_NOEXCEPT
    {
        return cudaExtent{width() * sizeof(T), height(), 1};
    }

    MUDA_GENERIC bool valid() const
    {
        return m_extent[0] != ~0 && m_extent[1] != ~0;
    }

    static MUDA_GENERIC Extent2D Zero() MUDA_NOEXCEPT { return Extent2D{0, 0}; }
};


class Extent3D
{
    size_t m_extent[3];

  public:
    MUDA_GENERIC Extent3D() MUDA_NOEXCEPT : m_extent{~0, ~0, ~0} {}
    MUDA_GENERIC Extent3D(size_t depth, size_t height, size_t width) MUDA_NOEXCEPT
        : m_extent{depth, height, width}
    {
    }

    MUDA_GENERIC size_t depth() const MUDA_NOEXCEPT { return m_extent[0]; }
    MUDA_GENERIC size_t height() const MUDA_NOEXCEPT { return m_extent[1]; }
    MUDA_GENERIC size_t width() const MUDA_NOEXCEPT { return m_extent[2]; }

    template <typename T>
    MUDA_GENERIC cudaExtent cuda_extent() const MUDA_NOEXCEPT
    {
        return cudaExtent{width() * sizeof(T), height(), depth()};
    }

    MUDA_GENERIC bool valid() const
    {
        return m_extent[0] != ~0 && m_extent[1] != ~0 && m_extent[2] != ~0;
    }

    static MUDA_GENERIC Extent3D Zero() MUDA_NOEXCEPT
    {
        return Extent3D{0, 0, 0};
    }
};

class Offset2D
{
    size_t m_offset[2];

  public:
    MUDA_GENERIC Offset2D() MUDA_NOEXCEPT : m_offset{~0, ~0} {}

    static MUDA_GENERIC Offset2D Zero() MUDA_NOEXCEPT { return Offset2D{0, 0}; }

    MUDA_GENERIC Offset2D(size_t offset_in_height, size_t offset_in_width) MUDA_NOEXCEPT
        : m_offset{offset_in_height, offset_in_width}
    {
    }

    MUDA_GENERIC size_t offset_in_height() const MUDA_NOEXCEPT
    {
        return m_offset[0];
    }

    MUDA_GENERIC size_t offset_in_width() const MUDA_NOEXCEPT
    {
        return m_offset[1];
    }

    template <typename T>
    MUDA_GENERIC cudaPos cuda_pos() const MUDA_NOEXCEPT
    {
        return cudaPos{offset_in_width() * sizeof(T), offset_in_height(), 0};
    }
};

class Offset3D
{
    size_t m_offset[3];

  public:
    MUDA_GENERIC Offset3D() MUDA_NOEXCEPT : m_offset{~0, ~0, ~0} {}

    static MUDA_GENERIC Offset3D Zero() MUDA_NOEXCEPT
    {
        return Offset3D{0, 0, 0};
    }

    MUDA_GENERIC Offset3D(size_t offset_in_depth, size_t offset_in_height, size_t offset_in_width) MUDA_NOEXCEPT
        : m_offset{offset_in_depth, offset_in_height, offset_in_width}
    {
    }

    MUDA_GENERIC size_t offset_in_depth() const MUDA_NOEXCEPT
    {
        return m_offset[0];
    }

    MUDA_GENERIC size_t offset_in_height() const MUDA_NOEXCEPT
    {
        return m_offset[1];
    }

    MUDA_GENERIC size_t offset_in_width() const MUDA_NOEXCEPT
    {
        return m_offset[2];
    }

    template <typename T>
    MUDA_GENERIC cudaPos cuda_pos() const MUDA_NOEXCEPT
    {
        return cudaPos{offset_in_width() * sizeof(T), offset_in_height(), offset_in_depth()};
    }
};


MUDA_GENERIC Extent2D operator-(const Extent2D& lhs, const Offset2D& rhs)
{
    return Extent2D{lhs.height() - rhs.offset_in_height(),
                    lhs.width() - rhs.offset_in_width()};
}

MUDA_GENERIC Extent3D operator-(const Extent3D& lhs, const Offset3D& rhs)
{
    return Extent3D{lhs.depth() - rhs.offset_in_depth(),
                    lhs.height() - rhs.offset_in_height(),
                    lhs.width() - rhs.offset_in_width()};
}

MUDA_GENERIC Extent2D operator+(const Extent2D& lhs, const Offset2D& rhs)
{
    return Extent2D{lhs.height() + rhs.offset_in_height(),
                    lhs.width() + rhs.offset_in_width()};
}

MUDA_GENERIC Extent3D operator+(const Extent3D& lhs, const Offset3D& rhs)
{
    return Extent3D{lhs.depth() + rhs.offset_in_depth(),
                    lhs.height() + rhs.offset_in_height(),
                    lhs.width() + rhs.offset_in_width()};
}

MUDA_GENERIC Offset2D operator+(const Offset2D& lhs, const Offset2D& rhs)
{
    return Offset2D{lhs.offset_in_height() + rhs.offset_in_height(),
                    lhs.offset_in_width() + rhs.offset_in_width()};
}

MUDA_GENERIC Offset3D operator+(const Offset3D& lhs, const Offset3D& rhs)
{
    return Offset3D{lhs.offset_in_depth() + rhs.offset_in_depth(),
                    lhs.offset_in_height() + rhs.offset_in_height(),
                    lhs.offset_in_width() + rhs.offset_in_width()};
}

MUDA_INLINE Extent2D max(const Extent2D& lhs, const Extent2D& rhs)
{
    return Extent2D{std::max(lhs.height(), rhs.height()),
                    std::max(lhs.width(), rhs.width())};
}

MUDA_INLINE Extent3D max(const Extent3D& lhs, const Extent3D& rhs)
{
    return Extent3D{std::max(lhs.depth(), rhs.depth()),
                    std::max(lhs.height(), rhs.height()),
                    std::max(lhs.width(), rhs.width())};
}

#define MUDA_DEFINE_COMPARISON_OPERATOR(op)                                    \
    MUDA_GENERIC bool operator op(const Extent2D& lhs, const Extent2D& rhs)    \
    {                                                                          \
        return (lhs.height() op rhs.height()) && (lhs.width() op rhs.width()); \
    }                                                                          \
    MUDA_GENERIC bool operator op(const Extent3D& lhs, const Extent3D& rhs)    \
    {                                                                          \
        return (lhs.depth() op rhs.depth()) && (lhs.height() op rhs.height())  \
               && (lhs.width() op rhs.width());                                \
    }

MUDA_DEFINE_COMPARISON_OPERATOR(<=);
MUDA_DEFINE_COMPARISON_OPERATOR(<);
MUDA_DEFINE_COMPARISON_OPERATOR(==);

#undef MUDA_DEFINE_COMPARISON_OPERATOR
}  // namespace muda