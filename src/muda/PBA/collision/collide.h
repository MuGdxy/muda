#pragma once
#include <Eigen/Core>
#include "bounding_volume.h"
#include "../../muda_def.h"

namespace muda
{
class collide
{
  public:
    using vec3 = Eigen::Vector3f;
    MUDA_GENERIC static float distance2(const vec3& p, AABB b)
    {
        float sqDist = 0.0f;
#pragma unroll
        for(int i = 0; i < 3; i++)
        {
            // for each axis count any excess distance outside box extents
            float v = p[i];
            if(v < b.min[i])
                sqDist += (b.min[i] - v) * (b.min[i] - v);
            if(v > b.max[i])
                sqDist += (v - b.max[i]) * (v - b.max[i]);
        }
        return sqDist;
    }

    MUDA_GENERIC static float distance(const vec3& p, AABB b)
    {
        return ::sqrt(distance2(p, b));
    }

    MUDA_GENERIC static bool detect(const sphere& s, const AABB& b)
    {
        // Compute squared distance between sphere center and AABB
        // the sqrt(dist) is fine to use as well, but this is faster.
        float sqDist = distance2(s.o, b);

        // Sphere and AABB intersect if the (squared) distance between them is
        // less than the (squared) sphere radius.
        return sqDist <= s.r * s.r;
    }

    MUDA_GENERIC static bool detect(const sphere& lhs, const sphere& rhs)
    {
        float r = lhs.r + rhs.r;
        return (lhs.o - rhs.o).squaredNorm() <= r * r;
    }

    MUDA_GENERIC static bool detect(const AABB& l, const AABB& r)
    {
        Eigen::Vector3i c;
#pragma unroll
        for(int i = 0; i < 3; ++i)
            c[i] = l.min[i] <= r.max[i] && l.max[i] >= r.min[i];
        return c.all();
    }
};
}  // namespace muda