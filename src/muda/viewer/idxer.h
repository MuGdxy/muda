#pragma once
#include "mapper.h"

namespace muda
{
template <typename T, int Dim>
class idxerND;


template <typename T>
class idxerND<T, 0>
{
    T* data_;

  public:
    MUDA_GENERIC idxerND() noexcept
        : data_(nullptr)
    {
    }
    
    MUDA_GENERIC explicit idxerND(T* p) noexcept
        : data_(p)
    {
    }

    MUDA_GENERIC idxerND& operator=(const T& rhs) noexcept
    {
        check();
        *data_ = rhs;
        return *this;
    }

    MUDA_GENERIC T& operator()() noexcept
    {
        check();
        return *data_;
    }
    MUDA_GENERIC const T& operator()() const noexcept
    {
        check();
        return *data_;
    }
    MUDA_GENERIC T& operator*() noexcept
    {
        check();
        return *data_;
    }
    MUDA_GENERIC const T& operator*() const noexcept
    {
        check();
        return *data_;
    }
    MUDA_GENERIC operator T&() noexcept
    {
        check();
        return *data_;
    }
    MUDA_GENERIC operator const T&() const noexcept
    {
        check();
        return *data_;
    }
    MUDA_GENERIC T* operator->()
    {
        check();
        return data_;
    }
    MUDA_GENERIC const T* operator->() const noexcept
    {
        check();
        return data_;
    }

    MUDA_GENERIC T*       data() noexcept { return data_; }
    MUDA_GENERIC const T* data() const noexcept { return data_; }

  private:
    MUDA_GENERIC __forceinline__ void check() const noexcept
    {
        if constexpr(debugViewers)
            if(data_ == nullptr)
            {
                muda_kernel_printf("idxer0D: data_ is null\n");
                if constexpr(trapOnError)
                    trap();
            }
    }
};

template <typename T>
class idxerND<T, 1> : public mapper<1>
{
    T* data_;

  public:
    MUDA_GENERIC idxerND() noexcept
        : data_(nullptr)
    {
    }
    MUDA_GENERIC idxerND(T* p, int dimx) noexcept
        : idxerND(p, Eigen::Vector<int, 1>(dimx))
    {
    }
    MUDA_GENERIC idxerND(T* p, const Eigen::Vector<int, 1>& dim) noexcept
        : mapper(dim)
        , data_((T*)p)
    {
    }
    MUDA_GENERIC idxerND(T* p, const mapper& m) noexcept
        : mapper(m)
        , data_((T*)p)
    {
    }

    MUDA_GENERIC const T& operator()(int x) const noexcept
    {
        check();
        return data_[map(x)];
    }
    MUDA_GENERIC T& operator()(int x) noexcept
    {
        check();
        return data_[map(x)];
    }

    MUDA_GENERIC T*       data() noexcept { return data_; }
    MUDA_GENERIC const T* data() const noexcept { return data_; }

  private:
    MUDA_GENERIC __forceinline__ void check() const noexcept
    {
        if constexpr(debugViewers)
            if(data_ == nullptr)
            {
                muda_kernel_printf("idxer1D: data_ is null\n");
                if constexpr(trapOnError)
                    trap();
            }
    }
};

template <typename T>
class idxerND<T, 2> : public mapper<2>
{
    T* data_;

  public:
    MUDA_GENERIC idxerND() noexcept
        : data_(nullptr)
    {
    }
    MUDA_GENERIC idxerND(T* p, int dimx, int dimy) noexcept
        : idxerND(p, Eigen::Vector<int, 2>(dimx, dimy))
    {
    }
    MUDA_GENERIC idxerND(T* p, const Eigen::Vector<int, 2>& dim) noexcept
        : mapper(dim)
        , data_((T*)p)
    {
    }
    MUDA_GENERIC idxerND(T* p, const mapper& m) noexcept
        : mapper(m)
        , data_((T*)p)
    {
    }

    MUDA_GENERIC const T& operator()(int x, int y) const noexcept
    {
        check();
        return data_[map(x, y)];
    }
    MUDA_GENERIC T& operator()(int x, int y) noexcept
    {
        check();
        return data_[map(x, y)];
    }

    MUDA_GENERIC T*       data() noexcept { return data_; }
    MUDA_GENERIC const T* data() const noexcept { return data_; }

  private:
    MUDA_GENERIC __forceinline__ void check() const noexcept
    {
        if constexpr(debugViewers)
            if(data_ == nullptr)
            {
                muda_kernel_printf("idxer1D: data_ is null\n");
                if constexpr(trapOnError)
                    trap();
            }
    }
};

template <typename T>
class idxerND<T, 3> : public mapper<3>
{
    T* data_;

  public:
    MUDA_GENERIC idxerND() noexcept
        : data_(nullptr){};
    MUDA_GENERIC idxerND(T* p, int dimx, int dimy, int dimz) noexcept
        : idxerND(p, Eigen::Vector<int, 2>(dimx, dimy, dimz))
    {
    }
    MUDA_GENERIC idxerND(T* p, const Eigen::Vector<int, 3>& dim) noexcept
        : mapper(dim)
        , data_((T*)p)
    {
    }
    MUDA_GENERIC idxerND(T* p, const mapper& m) noexcept
        : mapper(m)
        , data_((T*)p)
    {
    }

    MUDA_GENERIC const T& operator()(int x, int y, int z) const noexcept
    {
        check();
        return data_[map(x, y, z)];
    }
    MUDA_GENERIC T& operator()(int x, int y, int z) noexcept
    {
        check();
        return data_[map(x, y, z)];
    }

    MUDA_GENERIC T*       data() noexcept { return data_; }
    MUDA_GENERIC const T* data() const noexcept { return data_; }

  private:
    MUDA_GENERIC __forceinline__ void check() const noexcept
    {
        if constexpr(debugViewers)
            if(data_ == nullptr)
            {
                muda_kernel_printf("idxer1D: data_ is null\n");
                if constexpr(trapOnError)
                    trap();
            }
    }
};

template <typename T>
using idxer = idxerND<T, 0>;

template <typename T>
using idxer1D = idxerND<T, 1>;

template <typename T>
using idxer2D = idxerND<T, 2>;

template <typename T>
using idxer3D = idxerND<T, 3>;
}  // namespace muda
