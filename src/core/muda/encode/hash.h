#pragma once
#include <Eigen/Core>
#include "../muda_def.h"

namespace muda
{
template <int xshift = 20, int yshift = 10, int zshift = 0>
class shift_hash
{
  public:
    //MUDA_GENERIC static uint32_t map(const Eigen::Vector3<T>& p, T cellsize)
    //{
    //    return ((uint32_t)(p.x() / cellsize) << xshift)
    //           | ((uint32_t)(p.y() / cellsize) << yshift)
    //           | ((uint32_t)(p.z() / cellsize) << zshift);
    //}

    MUDA_GENERIC static uint32_t map(const Eigen::Vector3<uint32_t>& p)
    {
        return (p.x() << xshift) | (p.y() << yshift) | (p.z() << zshift);
    }

    MUDA_GENERIC static uint32_t map(const Eigen::Vector3i& p)
    {
        return ((uint32_t)p.x() << xshift) | ((uint32_t)p.y() << yshift)
               | ((uint32_t)p.z() << zshift);
    }

    MUDA_GENERIC uint32_t operator()(const Eigen::Vector3<uint32_t>& p) const
    {
        return map(p);
    }

    MUDA_GENERIC uint32_t operator()(const Eigen::Vector3i& p) const
    {
        return map(p);
    }
};
}  // namespace muda