#pragma once
#include "base.h"

namespace muda
{
template <typename T, int Dim>
class denseND;


template <typename T>
class denseND<T, 0> : public viewer_base
{
    T* m_data;

  public:
    using value_type = T;

    MUDA_GENERIC denseND() noexcept
        : m_data(nullptr)
    {
    }

    MUDA_GENERIC explicit denseND(T* p) noexcept
        : m_data(p)
    {
    }

    MUDA_GENERIC denseND& operator=(const T& rhs) noexcept
    {
        check();
        *m_data = rhs;
        return *this;
    }

    MUDA_GENERIC T& operator()() noexcept
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC const T& operator()() const noexcept
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC T& operator*() noexcept
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC const T& operator*() const noexcept
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC operator T&() noexcept
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC operator const T&() const noexcept
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC T* operator->()
    {
        check();
        return m_data;
    }
    MUDA_GENERIC const T* operator->() const noexcept
    {
        check();
        return m_data;
    }

    MUDA_GENERIC T* data() noexcept { return m_data; }

    MUDA_GENERIC const T* data() const noexcept { return m_data; }

  private:
    MUDA_GENERIC __forceinline__ void check() const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense[%s]: m_data is null\n", name());
    }
};

template <typename T>
class denseND<T, 1> : public viewer_base
{
    T*                    m_data;
    Eigen::Vector<int, 1> m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC denseND() noexcept
        : m_data(nullptr)
    {
    }

    MUDA_GENERIC denseND(T* p, int dimx) noexcept
        : denseND(p, Eigen::Vector<int, 1>(dimx))
    {
    }

    MUDA_GENERIC denseND(T* p, const Eigen::Vector<int, 1>& dim) noexcept
        : m_data(p)
        , m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x) const noexcept
    {
        check();
        return m_data[map(x)];
    }

    MUDA_GENERIC T& operator()(int x) noexcept
    {
        check();
        return m_data[map(x)];
    }

    MUDA_GENERIC int map(int x) const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim[0]))
                muda_kernel_error(
                    "dense1D[%s]: out of range, index=(%d) m_dim=(%d)\n",name(), x, m_dim[0]);
        return x;
    }

    MUDA_GENERIC T* data() noexcept { return m_data; }

    MUDA_GENERIC const T* data() const noexcept { return m_data; }

    MUDA_GENERIC int total_size() const noexcept { return m_dim[0]; }

    MUDA_GENERIC int dim() const noexcept { return m_dim[0]; }

  private:
    MUDA_GENERIC __forceinline__ void check() const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense1D[%s]: m_data is null\n", name());
    }
};

template <typename T>
class denseND<T, 2> : public viewer_base
{
    T*                    m_data;
    Eigen::Vector<int, 2> m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC denseND() noexcept
        : m_data(nullptr)
    {
    }

    MUDA_GENERIC denseND(T* p, int dimx, int dimy) noexcept
        : denseND(p, Eigen::Vector<int, 2>(dimx, dimy))
    {
    }

    MUDA_GENERIC denseND(T* p, const Eigen::Vector<int, 2>& dim) noexcept
        : m_data((T*)p)
        , m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x, int y) const noexcept
    {
        check();
        return m_data[map(x, y)];
    }

    MUDA_GENERIC T& operator()(int x, int y) noexcept
    {
        check();
        return m_data[map(x, y)];
    }

    MUDA_GENERIC T* data() noexcept { return m_data; }

    MUDA_GENERIC const T* data() const noexcept { return m_data; }

    // map index (x,y) to an offset. offset = x * dim_y + y
    MUDA_GENERIC int map(int x, int y) const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim[0] && y >= 0 && y < m_dim[1]))
            {
                muda_kernel_error("dense2D[%s]: out of range, index=(%d,%d) dim=(%d,%d)\n",
                                  name(),
                                  x,
                                  y,
                                  m_dim[0],
                                  m_dim[1]);
            }
        return x * m_dim[1] + y;
    }
    MUDA_GENERIC int total_size() const noexcept { return m_dim[0] * m_dim[1]; }

    MUDA_GENERIC int area() const noexcept { return total_size(); }

    template <int i>
    MUDA_GENERIC int dim() const noexcept
    {
        static_assert(i >= 0 && i <= 2, "dense2D: dim index out of range");
        return m_dim[i];
    }

  private:
    MUDA_GENERIC __forceinline__ void check() const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense2D[%s]: m_data is null\n", name());
    }
};

template <typename T>
class denseND<T, 3> : public viewer_base
{
    T*                    m_data;
    Eigen::Vector<int, 3> m_dim;
    int                   m_area;

  public:
    using value_type = T;

    MUDA_GENERIC denseND() noexcept
        : m_data(nullptr){};

    MUDA_GENERIC denseND(T* p, int dimx, int dimy, int dimz) noexcept
        : denseND(p, Eigen::Vector<int, 3>(dimx, dimy, dimz))
    {
    }

    MUDA_GENERIC denseND(T* p, const Eigen::Vector<int, 3>& dim) noexcept
        : m_data(p)
        , m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x, int y, int z) const noexcept
    {
        check();
        return m_data[map(x, y, z)];
    }

    MUDA_GENERIC T& operator()(int x, int y, int z) noexcept
    {
        check();
        return m_data[map(x, y, z)];
    }

    MUDA_GENERIC T* data() noexcept { return m_data; }

    MUDA_GENERIC const T* data() const noexcept { return m_data; }

    // map index (x,y,z) to an offset. offset = dim_z * (x * dim_y + y) + z
    MUDA_GENERIC int map(int x, int y, int z) const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim[0] && y >= 0 && y < m_dim[1] && z >= 0 && z < m_dim[2]))
                muda_kernel_error("dense3D[%s]: out of range, index=(%d,%d,%d) dim=(%d,%d,%d)\n",
                                  name(),
                                  x,
                                  y,
                                  z,
                                  m_dim[0],
                                  m_dim[1],
                                  m_dim[2]);
        return x * m_area + y * m_dim[2] + z;
    }

    template <int i>
    MUDA_GENERIC int dim() const noexcept
    {
        static_assert(i >= 0 && i <= 2, "dense3D: dim index out of range");
        return m_dim[i];
    }

    MUDA_GENERIC int area() const noexcept { return m_area; }

    MUDA_GENERIC int total_size() const noexcept { return m_dim[0] * m_area; }

    MUDA_GENERIC int volume() const noexcept { return total_size(); }

  private:
    MUDA_GENERIC __forceinline__ void check() const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense3D[%s]: data is null\n", name());
    }
};

template <typename T>
using dense = denseND<T, 0>;

template <typename T>
using dense1D = denseND<T, 1>;

template <typename T>
using dense2D = denseND<T, 2>;

template <typename T>
using dense3D = denseND<T, 3>;
}  // namespace muda

namespace muda
{
template <typename T>
MUDA_GENERIC inline auto make_dense(T* data) noexcept
{
    return dense<T>(data);
}

template <typename T>
MUDA_GENERIC inline auto make_dense1D(T* data, int dimx) noexcept
{
    return dense1D<T>(data, dimx);
}

template <typename T>
MUDA_GENERIC inline auto make_dense2D(T* data, int dimx, int dimy) noexcept
{
    return dense2D<T>(data, dimx, dimy);
}

template <typename T>
MUDA_GENERIC inline auto make_dense2D(T* data, Eigen::Vector2i dim) noexcept
{
    return dense2D<T>(data, dim);
}

template <typename T>
MUDA_GENERIC inline auto make_dense3D(T* data, int dimx, int dimy, int dimz) noexcept
{
    return dense3D<T>(data, dimx, dimy, dimz);
}

template <typename T>
MUDA_GENERIC inline auto make_dense3D(T* data, Eigen::Vector3i dim) noexcept
{
    return dense3D<T>(data, dim);
}
}  // namespace muda