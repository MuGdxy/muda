#pragma once
#include <Eigen/Core>
#include "../muda_def.h"

namespace muda
{
template <typename T = float, int xshift = 20, int yshift = 10, int zshift = 0>
class hash
{
  public:
    MUDA_GENERIC static unsigned int map(const Eigen::Vector3<T>& p, T cellsize)
    {
        return ((unsigned int)(p.x() / cellsize) << xshift)
               | ((unsigned int)(p.y() / cellsize) << yshift)
               | ((unsigned int)(p.z() / cellsize) << zshift);
    }

    MUDA_GENERIC static unsigned int map(const Eigen::Vector3<unsigned int>& p)
    {
        return (p.x() << xshift) | (p.y() << yshift) | (p.z() << zshift);
    }

    MUDA_GENERIC static unsigned int map(const Eigen::Vector3i& p)
    {
        return ((unsigned int)p.x() << xshift) | ((unsigned int)p.y() << yshift)
               | ((unsigned int)p.z() << zshift);
    }
};
}  // namespace muda