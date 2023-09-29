#pragma once
#include "base.h"

namespace muda
{
template <typename T, int Dim>
class DenseND;

template <typename T, int Dim>
class CDenseND;

template <typename T, int Dim>
struct read_only_viewer<DenseND<T, Dim>>
{
    using type = CDenseND<T, Dim>;
};

template <typename T, int Dim>
struct read_write_viewer<CDenseND<T, Dim>>
{
    using type = DenseND<T, Dim>;
};

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
                MUDA_KERNEL_ERROR("dense[%s:%s]: m_data is null\n",
                                  this->name(),
                                  this->kernel_name());
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
    MUDA_GENERIC operator CDenseND<T, 0>() const MUDA_NOEXCEPT
    {
        return CDenseND<T, 0>(m_data);
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
                MUDA_KERNEL_ERROR("dense[%s:%s]: m_data is null\n",
                                  this->name(),
                                  this->kernel_name());
    }
};

template <typename T>
class CDenseND<T, 1> : public ROViewer
{
    MUDA_VIEWER_COMMON(CDenseND);
    const T* m_data;
    int      m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC CDenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC CDenseND(const T* p, int dim) MUDA_NOEXCEPT : m_data(p), m_dim(dim)
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
            if(!(x >= 0 && x < m_dim))
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: out of range, index=(%d) m_dim=(%d)\n",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  m_dim);
        return x;
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC int dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC this_type sub_view(int offset, int size = -1) MUDA_NOEXCEPT
    {
        if(size < 0)
            size = m_dim - offset;
        if constexpr(DEBUG_VIEWER)
        {
            if(offset < 0 || offset + size > m_dim)
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: sub_view out of range, offset=%d size=%d m_dim=(%d)\n",
                                  this->name(),
                                  this->kernel_name(),
                                  offset,
                                  size,
                                  m_dim);
        }
        return this_type(m_data + offset, size);
    }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: m_data is null\n",
                                  this->name(),
                                  this->kernel_name());
    }
};


template <typename T>
class DenseND<T, 1> : public RWViewer
{
    MUDA_VIEWER_COMMON(DenseND);
    T*  m_data;
    int m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC DenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC DenseND(T* p, int dim) MUDA_NOEXCEPT : m_data(p), m_dim(dim) {}

    MUDA_GENERIC operator CDenseND<T, 1>() const MUDA_NOEXCEPT
    {
        return CDenseND<T, 1>(m_data, m_dim);
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
            if(!(x >= 0 && x < m_dim))
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: out of range, index=(%d) m_dim=(%d)\n",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  m_dim);
        return x;
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC int dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC this_type sub_view(int offset, int size = -1) MUDA_NOEXCEPT
    {
        if(size < 0)
            size = m_dim - offset;
        if constexpr(DEBUG_VIEWER)
        {
            if(offset < 0 || offset + size > m_dim)
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: sub_view out of range, offset=%d size=%d m_dim=(%d)\n",
                                  this->name(),
                                  this->kernel_name(),
                                  offset,
                                  size,
                                  m_dim);
        }
        return this_type(m_data + offset, size);
    }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: m_data is null\n",
                                  this->name(),
                                  this->kernel_name());
    }
};


template <typename T>
class CDenseND<T, 2> : public ROViewer
{
    MUDA_VIEWER_COMMON(CDenseND);
    const T* m_data;
    int2     m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC CDenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC CDenseND(const T* p, int dimx, int dimy) MUDA_NOEXCEPT
        : CDenseND(p, make_int2(dimx, dimy))
    {
    }

    MUDA_GENERIC CDenseND(const T* p, const int2& dim) MUDA_NOEXCEPT : m_data((T*)p),
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
            if(!(x >= 0 && x < m_dim.x && y >= 0 && y < m_dim.y))
            {
                MUDA_KERNEL_ERROR("dense2D[%s:%s]: out of range, index=(%d,%d) dim=(%d,%d)\n",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  y,
                                  m_dim.x,
                                  m_dim.y);
            }
        return x * m_dim.y + y;
    }
    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim.x * m_dim.y;
    }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return total_size(); }

    MUDA_GENERIC auto dim() const MUDA_NOEXCEPT { return m_dim; }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("dense2D[%s:%s]: m_data is null\n",
                                  this->name(),
                                  this->kernel_name());
    }
};

template <typename T>
class DenseND<T, 2> : public RWViewer
{
    MUDA_VIEWER_COMMON(DenseND);
    T*   m_data;
    int2 m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC DenseND() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC DenseND(T* p, int dimx, int dimy) MUDA_NOEXCEPT
        : DenseND(p, make_int2(dimx, dimy))
    {
    }

    MUDA_GENERIC DenseND(T* p, const int2& dim) MUDA_NOEXCEPT : m_data((T*)p), m_dim(dim)
    {
    }

    MUDA_GENERIC operator CDenseND<T, 2>() const MUDA_NOEXCEPT
    {
        return CDenseND<T, 2>(m_data, m_dim);
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
            if(!(x >= 0 && x < m_dim.x && y >= 0 && y < m_dim.y))
            {
                MUDA_KERNEL_ERROR("dense2D[%s:%s]: out of range, index=(%d,%d) dim=(%d,%d)\n",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  y,
                                  m_dim.x,
                                  m_dim.y);
            }
        return x * m_dim.y + y;
    }
    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim.x * m_dim.y;
    }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return total_size(); }

    MUDA_GENERIC auto dim() const MUDA_NOEXCEPT { return m_dim; }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("dense2D[%s:%s]: m_data is null\n",
                                  this->name(),
                                  this->kernel_name());
    }
};

template <typename T>
class CDenseND<T, 3> : public ROViewer
{
    MUDA_VIEWER_COMMON(CDenseND);
    const T* m_data;
    int3     m_dim;
    int      m_area;

  public:
    using value_type = T;

    MUDA_GENERIC CDenseND() MUDA_NOEXCEPT : m_data(nullptr){};

    MUDA_GENERIC CDenseND(const T* p, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
        : CDenseND(p, make_int3(dimx, dimy, dimz))
    {
    }

    MUDA_GENERIC CDenseND(const T* p, const int3& dim) MUDA_NOEXCEPT : m_data(p),
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
            if(!(x >= 0 && x < m_dim.x && y >= 0 && y < m_dim.y && z >= 0
                 && z < m_dim.z))
                MUDA_KERNEL_ERROR("dense3D[%s:%s]: out of range, index=(%d,%d,%d) dim=(%d,%d,%d)\n",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  y,
                                  z,
                                  m_dim.x,
                                  m_dim.y,
                                  m_dim.z);
        return x * m_area + y * m_dim.z + z;
    }

    MUDA_GENERIC auto dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return m_area; }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim.x * m_area;
    }

    MUDA_GENERIC int volume() const MUDA_NOEXCEPT { return total_size(); }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("dense3D[%s:%s]: data is null\n",
                                  this->name(),
                                  this->kernel_name());
    }
};

template <typename T>
class DenseND<T, 3> : public RWViewer
{
    MUDA_VIEWER_COMMON(DenseND);
    T*   m_data;
    int3 m_dim;
    int  m_area;

  public:
    using value_type = T;

    MUDA_GENERIC DenseND() MUDA_NOEXCEPT : m_data(nullptr){};

    MUDA_GENERIC DenseND(T* p, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
        : DenseND(p, make_int3(dimx, dimy, dimz))
    {
    }

    MUDA_GENERIC DenseND(T* p, const int3& dim) MUDA_NOEXCEPT : m_data(p), m_dim(dim)
    {
    }

    MUDA_GENERIC operator CDenseND<T, 3>() const MUDA_NOEXCEPT
    {
        return CDenseND<T, 3>(m_data, m_dim);
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
            if(!(x >= 0 && x < m_dim.x && y >= 0 && y < m_dim.y && z >= 0
                 && z < m_dim.z))
                MUDA_KERNEL_ERROR("dense3D[%s:%s]: out of range, index=(%d,%d,%d) dim=(%d,%d,%d)\n",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  y,
                                  z,
                                  m_dim.x,
                                  m_dim.y,
                                  m_dim.z);
        return x * m_area + y * m_dim.z + z;
    }

    MUDA_GENERIC auto dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return m_area; }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim.x * m_area;
    }

    MUDA_GENERIC int volume() const MUDA_NOEXCEPT { return total_size(); }

  private:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("dense3D[%s:%s]: data is null\n",
                                  this->name(),
                                  this->kernel_name());
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
MUDA_INLINE MUDA_GENERIC auto make_cdense2D(const T* data, const int2& dim) MUDA_NOEXCEPT
{
    return CDense2D<T>(data, dim);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense2D(T* data, const int2& dim) MUDA_NOEXCEPT
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
MUDA_INLINE MUDA_GENERIC auto make_cdense3D(const T* data, const int3& dim) MUDA_NOEXCEPT
{
    return CDense3D<T>(data, dim);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense3D(T* data, const int3& dim) MUDA_NOEXCEPT
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
MUDA_INLINE MUDA_GENERIC auto make_cdense(const T* data, const int2& dim) MUDA_NOEXCEPT
{
    return make_cdense2D(data, dim.x, dim.y);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense(T* data, const int2& dim) MUDA_NOEXCEPT
{
    return make_dense2D(data, dim.x, dim.y);
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
MUDA_INLINE MUDA_GENERIC auto make_cdense(const T* data, const int3& dim) noexcept
{
    return make_cdense3D(data, dim);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense(T* data, const int3& dim) noexcept
{
    return make_dense3D(data, dim);
}
}  // namespace muda