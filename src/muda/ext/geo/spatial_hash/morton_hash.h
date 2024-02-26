#include <muda/muda_def.h>
#include <cinttypes>
#include <Eigen/Core>

namespace muda::spatial_hash
{
template <typename T>
class Morton
{
    static_assert("Morton not implemented for this type.");
};

template <>
class Morton<int32_t>
{
  public:
    MUDA_GENERIC int32_t operator()(Eigen::Vector3i p) const
    {
        return (*this)(p.x(), p.y(), p.z());
    }

    constexpr MUDA_GENERIC int32_t operator()(int32_t x, int32_t y, int32_t z) const
    {
        x = expand_bits(x);
        y = expand_bits(y);
        z = expand_bits(z);
        return x | y << 1 | z << 2;
    }

  private:
    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
    constexpr MUDA_GENERIC static int32_t expand_bits(int32_t v)
    {
        //v = (v * 0x00010001u) & 0xFF0000FFu;
        //v = (v * 0x00000101u) & 0x0F00F00Fu;
        //v = (v * 0x00000011u) & 0xC30C30C3u;
        //v = (v * 0x00000005u) & 0x49249249u;

        v &= 0x3ff;
        v = (v | v << 16) & 0x30000ff;
        v = (v | v << 8) & 0x300f00f;
        v = (v | v << 4) & 0x30c30c3;
        v = (v | v << 2) & 0x9249249;
        return (int32_t)v;
    }
};

template <>
class Morton<int64_t>
{
  public:
    MUDA_GENERIC int64_t operator()(Eigen::Vector3i p) const
    {
        return (*this)(p.x(), p.y(), p.z());
    }

    constexpr MUDA_GENERIC int64_t operator()(int32_t x, int32_t y, int32_t z) const
    {
        x = expand_bits(x);
        y = expand_bits(y);
        z = expand_bits(z);
        return x | y << 1 | z << 2;
    }

  private:
    // Expands a 21-bit integer into 63 bits
    // by inserting 2 zeros after each bit.
    constexpr MUDA_GENERIC static int64_t expand_bits(int64_t v)
    {
        v &= 0x1fffff;
        v = (v | v << 32) & 0x1f00000000ffff;
        v = (v | v << 16) & 0x1f0000ff0000ff;
        v = (v | v << 8) & 0x100f00f00f00f00f;
        v = (v | v << 4) & 0x10c30c30c30c30c3;
        v = (v | v << 2) & 0x1249249249249249;
        return v;
    }
};
}  // namespace muda::spatial_hash