#pragma once
#include "base.h"

namespace muda
{
template <typename T, int Dim>
class DenseND;

template <typename T, int Dim>
class CDenseND;

template <typename T>
class CDenseND<T, 0> : public ROViewer
{
    MUDA_VIEWER_COMMON(CDenseND);
    const T* m_data;

  public:
    using value_type = T;

    MUDA_GENERIC CDenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC explicit CDenseND(const T* p) MUDA_NOEXCEPT : m_data(p) {}

    MUDA_GENERIC const T& operator()() const MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }

    MUDA_GENERIC const T& operator*() const MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }

    MUDA_GENERIC operator const T&() const MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }

    MUDA_GENERIC const T* operator->() const MUDA_NOEXCEPT
    {
        check();
        return m_data;
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense[%s]: m_data is null\n", this->name());
    }
};

template <typename T>
class DenseND<T, 0> : public RWViewer
{
    MUDA_VIEWER_COMMON(DenseND);
    T* m_data;

  public:
    using value_type = T;

    MUDA_GENERIC DenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC explicit DenseND(T* p) MUDA_NOEXCEPT : m_data(p) {}

    MUDA_GENERIC DenseND& operator=(const T& rhs) MUDA_NOEXCEPT
    {
        check();
        *m_data = rhs;
        return *this;
    }

    MUDA_GENERIC T& operator()() MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC const T& operator()() const MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC T& operator*() MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC const T& operator*() const MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC operator T&() MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC operator const T&() const MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }
    MUDA_GENERIC T* operator->()
    {
        check();
        return m_data;
    }
    MUDA_GENERIC const T* operator->() const MUDA_NOEXCEPT
    {
        check();
        return m_data;
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense[%s]: m_data is null\n", this->name());
    }
};

template <typename T>
class CDenseND<T, 1> : public ROViewer
{
    MUDA_VIEWER_COMMON(CDenseND);
    const T*              m_data;
    Eigen::Vector<int, 1> m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC CDenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC CDenseND(const T* p, int dimx) MUDA_NOEXCEPT
        : CDenseND(p, Eigen::Vector<int, 1>(dimx))
    {
    }

    MUDA_GENERIC CDenseND(const T* p, const Eigen::Vector<int, 1>& dim) MUDA_NOEXCEPT
        : m_data(p),
          m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x) const MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x)];
    }

    MUDA_GENERIC int map(int x) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim[0]))
                muda_kernel_error("dense1D[%s]: out of range, index=(%d) m_dim=(%d)\n",
                                  this->name(),
                                  x,
                                  m_dim[0]);
        return x;
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT { return m_dim[0]; }

    MUDA_GENERIC int dim() const MUDA_NOEXCEPT { return m_dim[0]; }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense1D[%s]: m_data is null\n", this->name());
    }
};


template <typename T>
class DenseND<T, 1> : public RWViewer
{
    MUDA_VIEWER_COMMON(DenseND);
    T*                    m_data;
    Eigen::Vector<int, 1> m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC DenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC DenseND(T* p, int dimx) MUDA_NOEXCEPT
        : DenseND(p, Eigen::Vector<int, 1>(dimx))
    {
    }

    MUDA_GENERIC DenseND(T* p, const Eigen::Vector<int, 1>& dim) MUDA_NOEXCEPT
        : m_data(p),
          m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x) const MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x)];
    }

    MUDA_GENERIC T& operator()(int x) MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x)];
    }

    MUDA_GENERIC int map(int x) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim[0]))
                muda_kernel_error("dense1D[%s]: out of range, index=(%d) m_dim=(%d)\n",
                                  this->name(),
                                  x,
                                  m_dim[0]);
        return x;
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT { return m_dim[0]; }

    MUDA_GENERIC int dim() const MUDA_NOEXCEPT { return m_dim[0]; }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense1D[%s]: m_data is null\n", this->name());
    }
};


template <typename T>
class CDenseND<T, 2> : public ROViewer
{
    MUDA_VIEWER_COMMON(CDenseND);
    const T*              m_data;
    Eigen::Vector<int, 2> m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC CDenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC CDenseND(const T* p, int dimx, int dimy) MUDA_NOEXCEPT
        : CDenseND(p, Eigen::Vector<int, 2>(dimx, dimy))
    {
    }

    MUDA_GENERIC CDenseND(const T* p, const Eigen::Vector<int, 2>& dim) MUDA_NOEXCEPT
        : m_data((T*)p),
          m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x, int y) const MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x, y)];
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    // map index (x,y) to an offset. offset = x * dim_y + y
    MUDA_GENERIC int map(int x, int y) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim[0] && y >= 0 && y < m_dim[1]))
            {
                muda_kernel_error("dense2D[%s]: out of range, index=(%d,%d) dim=(%d,%d)\n",
                                  this->name(),
                                  x,
                                  y,
                                  m_dim[0],
                                  m_dim[1]);
            }
        return x * m_dim[1] + y;
    }
    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim[0] * m_dim[1];
    }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return total_size(); }

    template <int i>
    MUDA_GENERIC int dim() const MUDA_NOEXCEPT
    {
        static_assert(i >= 0 && i <= 2, "dense2D: dim index out of range");
        return m_dim[i];
    }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense2D[%s]: m_data is null\n", this->name());
    }
};

template <typename T>
class DenseND<T, 2> : public RWViewer
{
    MUDA_VIEWER_COMMON(DenseND);
    T*                    m_data;
    Eigen::Vector<int, 2> m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC DenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC DenseND(T* p, int dimx, int dimy) MUDA_NOEXCEPT
        : DenseND(p, Eigen::Vector<int, 2>(dimx, dimy))
    {
    }

    MUDA_GENERIC DenseND(T* p, const Eigen::Vector<int, 2>& dim) MUDA_NOEXCEPT
        : m_data((T*)p),
          m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x, int y) const MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x, y)];
    }

    MUDA_GENERIC T& operator()(int x, int y) MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x, y)];
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    // map index (x,y) to an offset. offset = x * dim_y + y
    MUDA_GENERIC int map(int x, int y) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim[0] && y >= 0 && y < m_dim[1]))
            {
                muda_kernel_error("dense2D[%s]: out of range, index=(%d,%d) dim=(%d,%d)\n",
                                  this->name(),
                                  x,
                                  y,
                                  m_dim[0],
                                  m_dim[1]);
            }
        return x * m_dim[1] + y;
    }
    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim[0] * m_dim[1];
    }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return total_size(); }

    template <int i>
    MUDA_GENERIC int dim() const MUDA_NOEXCEPT
    {
        static_assert(i >= 0 && i <= 2, "dense2D: dim index out of range");
        return m_dim[i];
    }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense2D[%s]: m_data is null\n", this->name());
    }
};

template <typename T>
class CDenseND<T, 3> : public ROViewer
{
    MUDA_VIEWER_COMMON(CDenseND);
    const T*              m_data;
    Eigen::Vector<int, 3> m_dim;
    int                   m_area;

  public:
    using value_type = T;

    MUDA_GENERIC CDenseND() MUDA_NOEXCEPT : m_data(nullptr){};

    MUDA_GENERIC CDenseND(const T* p, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
        : CDenseND(p, Eigen::Vector<int, 3>(dimx, dimy, dimz))
    {
    }

    MUDA_GENERIC CDenseND(const T* p, const Eigen::Vector<int, 3>& dim) MUDA_NOEXCEPT
        : m_data(p),
          m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x, int y, int z) const MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x, y, z)];
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    // map index (x,y,z) to an offset. offset = dim_z * (x * dim_y + y) + z
    MUDA_GENERIC int map(int x, int y, int z) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim[0] && y >= 0 && y < m_dim[1] && z >= 0 && z < m_dim[2]))
                muda_kernel_error("dense3D[%s]: out of range, index=(%d,%d,%d) dim=(%d,%d,%d)\n",
                                  this->name(),
                                  x,
                                  y,
                                  z,
                                  m_dim[0],
                                  m_dim[1],
                                  m_dim[2]);
        return x * m_area + y * m_dim[2] + z;
    }

    template <int i>
    MUDA_GENERIC int dim() const MUDA_NOEXCEPT
    {
        static_assert(i >= 0 && i <= 2, "dense3D: dim index out of range");
        return m_dim[i];
    }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return m_area; }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim[0] * m_area;
    }

    MUDA_GENERIC int volume() const MUDA_NOEXCEPT { return total_size(); }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense3D[%s]: data is null\n", this->name());
    }
};

template <typename T>
class DenseND<T, 3> : public RWViewer
{
    MUDA_VIEWER_COMMON(DenseND);
    T*                    m_data;
    Eigen::Vector<int, 3> m_dim;
    int                   m_area;

  public:
    using value_type = T;

    MUDA_GENERIC DenseND() MUDA_NOEXCEPT : m_data(nullptr){};

    MUDA_GENERIC DenseND(T* p, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
        : DenseND(p, Eigen::Vector<int, 3>(dimx, dimy, dimz))
    {
    }

    MUDA_GENERIC DenseND(T* p, const Eigen::Vector<int, 3>& dim) MUDA_NOEXCEPT
        : m_data(p),
          m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x, int y, int z) const MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x, y, z)];
    }

    MUDA_GENERIC T& operator()(int x, int y, int z) MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x, y, z)];
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    // map index (x,y,z) to an offset. offset = dim_z * (x * dim_y + y) + z
    MUDA_GENERIC int map(int x, int y, int z) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim[0] && y >= 0 && y < m_dim[1] && z >= 0 && z < m_dim[2]))
                muda_kernel_error("dense3D[%s]: out of range, index=(%d,%d,%d) dim=(%d,%d,%d)\n",
                                  this->name(),
                                  x,
                                  y,
                                  z,
                                  m_dim[0],
                                  m_dim[1],
                                  m_dim[2]);
        return x * m_area + y * m_dim[2] + z;
    }

    template <int i>
    MUDA_GENERIC int dim() const MUDA_NOEXCEPT
    {
        static_assert(i >= 0 && i <= 2, "dense3D: dim index out of range");
        return m_dim[i];
    }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return m_area; }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim[0] * m_area;
    }

    MUDA_GENERIC int volume() const MUDA_NOEXCEPT { return total_size(); }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                muda_kernel_error("dense3D[%s]: data is null\n", this->name());
    }
};

template <typename T>
using CDense = CDenseND<T, 0>;

template <typename T>
using Dense = DenseND<T, 0>;

template <typename T>
using CDense1D = CDenseND<T, 1>;

template <typename T>
using Dense1D = DenseND<T, 1>;

template <typename T>
using CDense2D = CDenseND<T, 2>;

template <typename T>
using Dense2D = DenseND<T, 2>;

template <typename T>
using CDense3D = CDenseND<T, 3>;

template <typename T>
using Dense3D = DenseND<T, 3>;
}  // namespace muda

namespace muda
{

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense(const T* data) MUDA_NOEXCEPT
{
    return CDense<T>(data);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense(T* data) MUDA_NOEXCEPT
{
    return Dense<T>(data);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense1D(const T* data, int dimx) MUDA_NOEXCEPT
{
    return CDense1D<T>(data, dimx);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense1D(T* data, int dimx) MUDA_NOEXCEPT
{
    return Dense1D<T>(data, dimx);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense2D(const T* data, int dimx, int dimy) MUDA_NOEXCEPT
{
    return CDense2D<T>(data, dimx, dimy);
}


template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense2D(T* data, int dimx, int dimy) MUDA_NOEXCEPT
{
    return Dense2D<T>(data, dimx, dimy);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense2D(const T* data, const Eigen::Vector2i& dim) MUDA_NOEXCEPT
{
    return CDense2D<T>(data, dim);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense2D(T* data, const Eigen::Vector2i& dim) MUDA_NOEXCEPT
{
    return Dense2D<T>(data, dim);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense3D(const T* data, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return CDense3D<T>(data, dimx, dimy, dimz);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense3D(T* data, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return Dense3D<T>(data, dimx, dimy, dimz);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense3D(const T* data, const Eigen::Vector3i& dim) MUDA_NOEXCEPT
{
    return CDense3D<T>(data, dim);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense3D(T* data, const Eigen::Vector3i& dim) MUDA_NOEXCEPT
{
    return Dense3D<T>(data, dim);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense(const T* data, int dimx) MUDA_NOEXCEPT
{
    return make_cdense1D(data, dimx);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense(T* data, int dimx) MUDA_NOEXCEPT
{
    return make_dense1D(data, dimx);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense(const T* data, int dimx, int dimy) MUDA_NOEXCEPT
{
    return make_cdense2D(data, dimx, dimy);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense(T* data, int dimx, int dimy) MUDA_NOEXCEPT
{
    return make_dense2D(data, dimx, dimy);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense(const T* data, const Eigen::Vector2i& dim) MUDA_NOEXCEPT
{
    return make_cdense2D(data, dim.x(), dim.y());
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense(T* data, const Eigen::Vector2i& dim) MUDA_NOEXCEPT
{
    return make_dense2D(data, dim.x(), dim.y());
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense(const T* data, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return make_cdense3D(data, dimx, dimy, dimz);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense(T* data, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return make_dense3D(data, dimx, dimy, dimz);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense(const T* data, const Eigen::Vector3i& dim) noexcept
{
    return make_cdense3D(data, dim);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense(T* data, const Eigen::Vector3i& dim) noexcept
{
    return make_dense3D(data, dim);
}
}  // namespace muda