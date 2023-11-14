#pragma once
#include <muda/viewer/viewer_base.h>

namespace muda
{
/*****************************************************************************************
 *
 * Dense2D (3D array)
 * indexing (x,y,z)
 *  1) non-pitched:  x * dim_y * dimz + y * dim_z + z 
 *  2) pitched:      x * dim_y * pitch + y * pitch + z 
 *
 * Note:
 *  1) z moves faster than y, y moves faster than x, which is the same as C/C++ 2d array
 *  2) as for CUDA Memory3D, x index into depth, y index into height, z index into width
 ****************************************************************************************/

template <typename T>
class Dense3DBase : public ViewerBase
{
  protected:
    T*   m_data;
    int3 m_dim;
    int  m_pitch;
    int  m_pitched_area;

  public:
    using value_type = T;

    MUDA_GENERIC Dense3DBase() MUDA_NOEXCEPT : m_data(nullptr){};

    MUDA_GENERIC Dense3DBase(T* p, const int3& dim, int pitch) MUDA_NOEXCEPT
        : m_data(p),
          m_dim(dim),
          m_pitch(pitch),
          m_pitched_area(dim.y* pitch)
    {
    }

    MUDA_GENERIC Dense3DBase(T* p, const int3& dim) MUDA_NOEXCEPT
        : Dense3DBase(p, dim, dim.z)
    {
    }

    MUDA_GENERIC const T& operator()(const int3& xyz) const MUDA_NOEXCEPT
    {
        return operator()(xyz.x, xyz.y, xyz.z);
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
                MUDA_KERNEL_ERROR("dense3D[%s:%s]: out of range, index=(%d,%d,%d) dim=(%d,%d,%d)",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  y,
                                  z,
                                  m_dim.x,
                                  m_dim.y,
                                  m_dim.z);
        return x * m_pitched_area + y * m_pitch + z;
    }

    MUDA_GENERIC auto dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return m_dim.y * m_dim.z; }

    MUDA_GENERIC int pitch() const MUDA_NOEXCEPT { return m_pitch; }

    MUDA_GENERIC int pitched_area() const MUDA_NOEXCEPT
    {
        return m_pitched_area;
    }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim.x * area();
    }

    MUDA_GENERIC int volume() const MUDA_NOEXCEPT { return total_size(); }

    MUDA_GENERIC int pitched_volume() const MUDA_NOEXCEPT
    {
        return m_dim.x * m_pitched_area;
    }

  protected:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("dense3D[%s:%s]: data is null",
                                  this->name(),
                                  this->kernel_name());
    }
};

template <typename T>
class CDense3D : public Dense3DBase<T>
{
    MUDA_VIEWER_COMMON_NAME(CDense3D);

  public:
    using Dense3DBase<T>::operator();

    MUDA_GENERIC CDense3D(const Dense3DBase<T>& base) MUDA_NOEXCEPT
        : Dense3DBase<T>(base){};

    MUDA_GENERIC CDense3D(const T* p, const int3& dim, int pitch) MUDA_NOEXCEPT
        : Dense3DBase<T>(const_cast<T*>(p), dim, pitch)
    {
    }

    MUDA_GENERIC CDense3D(const T* p, const int3& dim) MUDA_NOEXCEPT
        : Dense3DBase<T>(const_cast<T*>(p), dim)
    {
    }
};

template <typename T>
class Dense3D : public Dense3DBase<T>
{
    MUDA_VIEWER_COMMON_NAME(Dense3D);

  public:
    using Dense3DBase<T>::Dense3DBase;
    using Dense3DBase<T>::operator();
    using Dense3DBase<T>::data;

    MUDA_GENERIC Dense3D(const Dense3DBase<T>& base) MUDA_NOEXCEPT : Dense3DBase<T>(base)
    {
    }

    MUDA_GENERIC operator CDense3D<T>() const MUDA_NOEXCEPT
    {
        return CDense3D<T>{*this};
    }

    MUDA_GENERIC T& operator()(const int3& xyz) MUDA_NOEXCEPT
    {
        return operator()(xyz.x, xyz.y, xyz.z);
    }

    MUDA_GENERIC T& operator()(int x, int y, int z) MUDA_NOEXCEPT
    {
        this->check();
        return m_data[map(x, y, z)];
    }
};

// viewer traits
template <typename T>
struct read_only_viewer<Dense3D<T>>
{
    using type = CDense3D<T>;
};

template <typename T>
struct read_write_viewer<CDense3D<T>>
{
    using type = Dense3D<T>;
};

// CTAD
template <typename T>
CDense3D(const T*, const int3&, int) -> CDense3D<T>;

template <typename T>
Dense3D(T*, const int3&, int) -> Dense3D<T>;

// make functions
template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense3D(const T* data, int dimx, int dimy, int dimz, int pitch) MUDA_NOEXCEPT
{
    return CDense3D<T>{data, make_int3(dimx, dimy, dimz), pitch};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense3D(const T* data, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return CDense3D<T>{data, make_int3(dimx, dimy, dimz)};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense3D(T* data, int dimx, int dimy, int dimz, int pitch) MUDA_NOEXCEPT
{
    return Dense3D<T>{data, make_int3(dimx, dimy, dimz), pitch};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense3D(T* data, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return Dense3D<T>{data, make_int3(dimx, dimy, dimz)};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense3D(const T* data, const int3& dim, int pitch) MUDA_NOEXCEPT
{
    return CDense3D<T>{data, dim, pitch};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense3D(const T* data, const int3& dim) MUDA_NOEXCEPT
{
    return CDense3D<T>{data, dim};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense3D(T* data, const int3& dim, int pitch) MUDA_NOEXCEPT
{
    return Dense3D<T>{data, dim, pitch};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense3D(T* data, const int3& dim) MUDA_NOEXCEPT
{
    return Dense3D<T>{data, dim};
}
}  // namespace muda