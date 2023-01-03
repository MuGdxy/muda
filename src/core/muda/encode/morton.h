#pragma once
#include <numeric>
#include <Eigen/Core>
#include "../muda_config.h"
#include "../muda_def.h"


namespace muda
{
class morton
{
  public:
    MUDA_GENERIC static uint32_t map(uint32_t x, uint32_t y, uint32_t z)
    {
        uint32_t xx = expand_bits((uint32_t)x);
        uint32_t yy = expand_bits((uint32_t)y);
        uint32_t zz = expand_bits((uint32_t)z);
        return xx << 2 + yy << 1 + zz;
    }

    MUDA_GENERIC uint32_t operator()(Eigen::Vector3<uint32_t> p) const
    {
        return map(p.x(), p.y(), p.z());
    }

  private:
    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
    MUDA_GENERIC static uint32_t expand_bits(uint32_t v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }
};
}  // namespace muda