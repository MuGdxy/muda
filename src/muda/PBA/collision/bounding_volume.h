#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../../muda_def.h"
#include "../../math/math.h"

namespace muda
{
class sphere
{
    using vec3 = Eigen::Vector3f;

  public:
    MUDA_GENERIC sphere(const Eigen::Vector3f& o, float r, uint32_t id = uint32_t(-1))
        : o(o)
        , r(r)
        , id(id)
    {
    }
    MUDA_GENERIC sphere()
        : o(vec3::Zero())
        , r(0.0f)
        , id(-1)
    {
    }
    uint32_t        id = -1;
    Eigen::Vector3f o;
    float           r;
};

class AABB
{
    using vec3 = Eigen::Vector3f;

  public:
    uint32_t id = -1;
    vec3     max;
    vec3     min;

    MUDA_GENERIC AABB(const vec3& x0, const vec3& x1, float radius = 0.0f)
    {
#pragma unroll
        for(int i = 0; i < 3; ++i)
        {
            max(i) = ::muda::max(x0(i) + radius, x1(i) + radius);
            min(i) = ::muda::min(x0(i) - radius, x1(i) - radius);
        }
    }

    MUDA_GENERIC AABB(const AABB& l, const AABB& r)
    {
        max = Max(l.max, r.max);
        min = Min(l.min, r.min);
    }

    MUDA_GENERIC AABB(const vec3& x0, const vec3& x1, const vec3& x0t, const vec3& x1t)
        : AABB(AABB(x0, x1), AABB(x0t, x1t))
    {
    }

    MUDA_GENERIC vec3 center() const { return (max + min) / 2; }

    MUDA_GENERIC float radius() const { return (max - min).norm() / 2; }
};
}  // namespace muda
