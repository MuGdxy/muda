#pragma once
#include <muda/viewer/viewer_base.h>

namespace muda
{
/*****************************************************************************
 *
 * Dense2D (2D array)
 * indexing (x,y)
 *  1) non-pitched:  x * dim_y + y 
 *  2) pitched:      x * pitch + y
 *
 * Note:
 *  1) y moves faster than x, which is the same as C/C++ 2d array
 *  2) as for CUDA Memory2D, x index into height, y index into width.
 *****************************************************************************/

template <typename T>
class Dense2DBase : public ViewerBase
{
  protected:
    T*   m_data;
    int2 m_dim;
    int  m_pitch;

  public:
    using value_type = T;

    MUDA_GENERIC Dense2DBase() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC Dense2DBase(T* p, const int2& dim, int pitch) MUDA_NOEXCEPT
        : m_data(p),
          m_dim(dim),
          m_pitch(pitch)
    {
    }

    MUDA_GENERIC Dense2DBase(T* p, const int2& dim) MUDA_NOEXCEPT
        : Dense2DBase(p, dim, dim.y)
    {
    }

    MUDA_GENERIC const T& operator()(const int2& xy) const MUDA_NOEXCEPT
    {
        return operator()(xy.x, xy.y);
    }

    MUDA_GENERIC const T& operator()(int x, int y) const MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x, y)];
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC auto total_size() const MUDA_NOEXCEPT
    {
        return m_dim.x * m_dim.y;
    }

    MUDA_GENERIC auto area() const MUDA_NOEXCEPT { return total_size(); }

    MUDA_GENERIC auto dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC auto pitch() const MUDA_NOEXCEPT { return m_pitch; }

    MUDA_GENERIC auto pitched_area() const MUDA_NOEXCEPT
    {
        return m_dim.x * m_pitch;
    }

    // map index (x,y) to an offset. offset = x * dim_y + y
    MUDA_GENERIC auto map(int x, int y) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim.x && y >= 0 && y < m_dim.y))
            {
                MUDA_KERNEL_ERROR("dense2D[%s:%s]: out of range, index=(%d,%d) dim=(%d,%d)",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  y,
                                  m_dim.x,
                                  m_dim.y);
            }
        return x * m_pitch + y;
    }

  protected:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("dense2D[%s:%s]: m_data is null",
                                  this->name(),
                                  this->kernel_name());
    }
};


template <typename T>
class CDense2D : public Dense2DBase<T>
{
    MUDA_VIEWER_COMMON_NAME(CDense2D);

  public:
    using Dense2DBase<T>::Dense2DBase;
    using Dense2DBase<T>::operator();

    MUDA_GENERIC CDense2D(const Dense2DBase<T>& base)
        : Dense2DBase<T>(base)
    {
    }

    MUDA_GENERIC CDense2D(const T* p, const int2& dim, int pitch) MUDA_NOEXCEPT
        : Dense2DBase<T>(const_cast<T*>(p), dim, pitch)
    {
    }

    MUDA_GENERIC CDense2D(const T* p, const int2& dim) MUDA_NOEXCEPT
        : Dense2DBase<T>(const_cast<T*>(p), dim)
    {
    }
};

template <typename T>
class Dense2D : public Dense2DBase<T>
{
    MUDA_VIEWER_COMMON_NAME(Dense2D);

  public:
    using Dense2DBase<T>::Dense2DBase;
    using Dense2DBase<T>::operator();
    using Dense2DBase<T>::data;

    MUDA_GENERIC Dense2D(Dense2DBase<T>& base)
        : Dense2DBase<T>(base)
    {
    }

    MUDA_GENERIC operator CDense2D<T>() const MUDA_NOEXCEPT
    {
        return CDense2D<T>{*this};
    }

    MUDA_GENERIC T& operator()(int x, int y) MUDA_NOEXCEPT
    {
        this->check();
        return this->m_data[this->map(x, y)];
    }

    MUDA_GENERIC T& operator()(const int2& xy) MUDA_NOEXCEPT
    {
        return this->operator()(xy.x, xy.y);
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT { return this->m_data; }
};

// viewer traits
template <typename T>
struct read_only_viewer<Dense2D<T>>
{
    using type = CDense2D<T>;
};

template <typename T>
struct read_write_viewer<CDense2D<T>>
{
    using type = Dense2D<T>;
};

// CTAD
template <typename T>
CDense2D(T*, const int2&, int) -> CDense2D<T>;
template <typename T>
Dense2D(T*, const int2&, int) -> Dense2D<T>;

// make functions
template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense2D(const T* data, int dimx, int dimy, int pitch) MUDA_NOEXCEPT
{
    return CDense2D<T>{data, make_int2(dimx, dimy), pitch};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense2D(const T* data, int dimx, int dimy) MUDA_NOEXCEPT
{
    return CDense2D<T>{data, make_int2(dimx, dimy)};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense2D(T* data, int dimx, int dimy, int pitch) MUDA_NOEXCEPT
{
    return Dense2D<T>{data, make_int2(dimx, dimy), pitch};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense2D(T* data, int dimx, int dimy) MUDA_NOEXCEPT
{
    return Dense2D<T>{data, make_int2(dimx, dimy)};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense2D(const T* data, const int2& dim, int pitch) MUDA_NOEXCEPT
{
    return CDense2D<T>{data, dim, pitch};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense2D(const T* data, const int2& dim) MUDA_NOEXCEPT
{
    return CDense2D<T>{data, dim};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense2D(T* data, const int2& dim, int pitch) MUDA_NOEXCEPT
{
    return Dense2D<T>{data, dim, pitch};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense2D(T* data, const int2& dim) MUDA_NOEXCEPT
{
    return Dense2D<T>{data, dim};
}
}  // namespace muda