#pragma once
#include "base.h"

namespace muda
{
template <int Dim>
class mapper;

template <>
class mapper<1>
{
  protected:
    Eigen::Vector<int, 1> dim_;

  public:
    MUDA_GENERIC mapper() = default;
    MUDA_GENERIC mapper(int dimx) noexcept
        : mapper(Eigen::Vector<int, 1>(dimx))
    {
    }
    MUDA_GENERIC mapper(const Eigen::Vector<int, 1>& dim) noexcept
        : dim_(dim)
    {
    }
    MUDA_GENERIC int map(int x) const noexcept
    {
        if constexpr(debugViewers)
            if(!(x >= 0 && x < dim_[0]))
                muda_kernel_error("mapper: out of range, index=(%d) dim_=(%d)\n", x, dim_[0]);
        return x;
    }
    MUDA_GENERIC int total_size() const noexcept { return dim_[0]; }
    MUDA_GENERIC int dim() const noexcept { return dim_[0]; }
};

template <>
class mapper<2>
{
  protected:
    Eigen::Vector<int, 2> dim_;

  public:
    MUDA_GENERIC mapper() = default;
    MUDA_GENERIC mapper(int dimx, int dimy) noexcept
        : mapper(Eigen::Vector<int, 2>(dimx, dimy))
    {
    }
    MUDA_GENERIC mapper(const Eigen::Vector<int, 2>& dim) noexcept
        : dim_(dim)
    {
    }
    MUDA_GENERIC int map(int x, int y) const noexcept
    {
        if constexpr(debugViewers)
            if(!(x >= 0 && x < dim_[0] && y >= 0 && y < dim_[1]))
            {
                muda_kernel_error("mapper: out of range, index=(%d,%d) dim_=(%d,%d)\n",
                                   x,
                                   y,
                                   dim_[0],
                                   dim_[1]);
            }
        return x * dim_[1] + y;
    }
    MUDA_GENERIC int total_size() const noexcept { return dim_[0] * dim_[1]; }
    MUDA_GENERIC int area() const noexcept { return total_size(); }

    template <int i>
    MUDA_GENERIC int dim() const noexcept
    {
        static_assert(i >= 0 && i <= 2, "out of range");
        return dim_[i];
    }
};

template <>
class mapper<3>
{
  protected:
    Eigen::Vector<int, 3> dim_;
    int                   area_;

  public:
    MUDA_GENERIC mapper() = default;
    MUDA_GENERIC mapper(int dimx, int dimy, int dimz) noexcept
        : mapper(Eigen::Vector<int, 3>(dimx, dimy, dimz))
    {
    }
    MUDA_GENERIC mapper(const Eigen::Vector<int, 3>& dim) noexcept
        : dim_(dim)
        , area_(dim[1] * dim[2])
    {
    }

    MUDA_GENERIC int map(int x, int y, int z) const noexcept
    {
        if constexpr(debugViewers)
            if(!(x >= 0 && x < dim_[0] && y >= 0 && y < dim_[1] && z >= 0 && z < dim_[2]))
                muda_kernel_error("mapper: out of range, index=(%d,%d,%d) dim_=(%d,%d,%d)\n",
                                   x,
                                   y,
                                   z,
                                   dim_[0],
                                   dim_[1],
                                   dim_[2]);
        return x * area_ + y * dim_[2] + z;
    }

    template <int i>
    MUDA_GENERIC int dim() const noexcept
    {
        static_assert(i >= 0 && i <= 2, "out of range");
        return dim_[i];
    }
    MUDA_GENERIC int area() const noexcept { return area_; }
    MUDA_GENERIC int volume() const noexcept { return total_size(); }
    MUDA_GENERIC int total_size() const noexcept { return dim_[0] * area_; }
};
}  // namespace muda