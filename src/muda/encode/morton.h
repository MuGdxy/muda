#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../muda_config.h"
#include <numeric>

namespace muda
{
template <typename T>
class morton
{
  public:
    // Calculates a 30-bit Morton code for the
    // given 3D point located within the unit cube [0,1].
    MUDA_GENERIC static unsigned int map(T x, T y, T z)
    {
        using namespace std;
        x = min(max(x * 1024.0f, 0.0f), 1023.0f);
        y = min(max(y * 1024.0f, 0.0f), 1023.0f);
        z = min(max(z * 1024.0f, 0.0f), 1023.0f);
        return map((unsigned int)x, (unsigned int)y, (unsigned int)z);
    }

    MUDA_GENERIC static unsigned int map(unsigned int x, unsigned int y, unsigned int z)
    {
        unsigned int xx = expand_bits((unsigned int)x);
        unsigned int yy = expand_bits((unsigned int)y);
        unsigned int zz = expand_bits((unsigned int)z);
        return xx << 2 + yy << 1 + zz;
    }

  private:
    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
    MUDA_GENERIC static unsigned int expand_bits(unsigned int v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }
};
}  // namespace muda