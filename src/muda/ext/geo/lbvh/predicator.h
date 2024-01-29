#pragma once
#include <muda/ext/geo/lbvh/aabb.h>

namespace muda::lbvh
{
template <typename Real>
struct query_overlap
{
    MUDA_GENERIC query_overlap(const AABB<Real>& tgt)
        : target(tgt)
    {
    }

    query_overlap()                                = default;
    ~query_overlap()                               = default;
    query_overlap(const query_overlap&)            = default;
    query_overlap(query_overlap&&)                 = default;
    query_overlap& operator=(const query_overlap&) = default;
    query_overlap& operator=(query_overlap&&)      = default;

    MUDA_GENERIC inline bool operator()(const AABB<Real>& box) noexcept
    {
        return intersects(box, target);
    }

    AABB<Real> target;
};

template <typename Real>
MUDA_GENERIC query_overlap<Real> overlaps(const AABB<Real>& region) noexcept
{
    return query_overlap<Real>(region);
}

template <typename Real>
struct query_nearest
{
    // float4/double4
    using vector_type = typename vector_of<Real>::type;

    MUDA_GENERIC query_nearest(const vector_type& tgt)
        : target(tgt)
    {
    }

    query_nearest()                                = default;
    ~query_nearest()                               = default;
    query_nearest(const query_nearest&)            = default;
    query_nearest(query_nearest&&)                 = default;
    query_nearest& operator=(const query_nearest&) = default;
    query_nearest& operator=(query_nearest&&)      = default;

    vector_type target;
};

MUDA_GENERIC inline query_nearest<float> nearest(const float4& point) noexcept
{
    return query_nearest<float>(point);
}
MUDA_GENERIC inline query_nearest<float> nearest(const float3& point) noexcept
{
    return query_nearest<float>(make_float4(point.x, point.y, point.z, 0.0f));
}
MUDA_GENERIC inline query_nearest<double> nearest(const double4& point) noexcept
{
    return query_nearest<double>(point);
}
MUDA_GENERIC inline query_nearest<double> nearest(const double3& point) noexcept
{
    return query_nearest<double>(make_double4(point.x, point.y, point.z, 0.0));
}

}  // namespace muda::lbvh
