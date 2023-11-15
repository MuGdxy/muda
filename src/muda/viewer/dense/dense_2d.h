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
    int2 m_offset;
    int2 m_dim;
    int  m_pitch_bytes;

  public:
    using value_type = T;

    MUDA_GENERIC Dense2DBase() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC Dense2DBase(T* p, const int2& offset, const int2& dim, int pitch_bytes) MUDA_NOEXCEPT
        : m_data(p),
          m_offset(offset),
          m_dim(dim),
          m_pitch_bytes(pitch_bytes)
    {
    }

    MUDA_GENERIC const T& operator()(const int2& xy) const MUDA_NOEXCEPT
    {
        return operator()(xy.x, xy.y);
    }

    MUDA_GENERIC const T& operator()(int x, int y) const MUDA_NOEXCEPT
    {
        check();
        check_range(x, y);

        x += m_offset.x;
        y += m_offset.y;
        auto height_begin = reinterpret_cast<std::byte*>(m_data) + x * m_pitch_bytes;
        return *(reinterpret_cast<T*>(height_begin) + y);
    }

    MUDA_GENERIC const T& flatten(int i)
    {
        if constexpr(DEBUG_VIEWER)
        {
            MUDA_KERNEL_ASSERT(i >= 0 && i < total_size(),
                               "dense2D[%s:%s]: out of range, index=%d, total_size=%d",
                               this->name(),
                               this->kernel_name(),
                               i,
                               total_size());
        }
        auto x = i / m_dim.y;
        auto y = i % m_dim.y;
        return operator()(x, y);
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC auto total_size() const MUDA_NOEXCEPT
    {
        return m_dim.x * m_dim.y;
    }

    MUDA_GENERIC auto area() const MUDA_NOEXCEPT { return total_size(); }

    MUDA_GENERIC auto dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC auto pitch_bytes() const MUDA_NOEXCEPT
    {
        return m_pitch_bytes;
    }

  protected:
    MUDA_INLINE MUDA_GENERIC void check_range(int x, int y) const MUDA_NOEXCEPT
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
    }

    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
        {
            MUDA_KERNEL_ASSERT(m_data,
                               "dense2D[%s:%s]: m_data is null",
                               this->name(),
                               this->kernel_name());
        }
    }
};


template <typename T>
class CDense2D : public Dense2DBase<T>
{
    MUDA_VIEWER_COMMON_NAME(CDense2D);
    using Base = Dense2DBase<T>;

  public:
    using Base::Base;
    using Base::operator();

    MUDA_GENERIC CDense2D(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CDense2D(const T* p, const int2& offset, const int2& dim, int pitch_bytes) MUDA_NOEXCEPT
        : Base(const_cast<T*>(p), offset, dim, pitch_bytes)
    {
    }
};

template <typename T>
class Dense2D : public Dense2DBase<T>
{
    MUDA_VIEWER_COMMON_NAME(Dense2D);
    using Base = Dense2DBase<T>;

  public:
    using Base::Dense2DBase;
    using Base::operator();
    using Base::data;
    using Base::flatten;

    MUDA_GENERIC Dense2D(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC operator CDense2D<T>() const MUDA_NOEXCEPT
    {
        return CDense2D<T>{*this};
    }

    MUDA_GENERIC T& operator()(int x, int y) MUDA_NOEXCEPT
    {
        return const_cast<T&>(Base::operator()(x, y));
    }

    MUDA_GENERIC T& operator()(const int2& xy) MUDA_NOEXCEPT
    {
        return const_cast<T&>(Base::operator()(xy));
    }

    MUDA_GENERIC T& flatten(int i) { return const_cast<T&>(Base::flatten(i)); }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT
    {
        return const_cast<T*>(Base::data());
    }
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

// make functions
template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense_2d(const T* data, const int2& dim) MUDA_NOEXCEPT
{
    return CDense2D<T>{data, make_int2(0, 0), dim, dim.y * sizeof(T)};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense_2d(T* data, const int2& dim) MUDA_NOEXCEPT
{
    return Dense2D<T>{data, make_int2(0, 0), dim, dim.y * sizeof(T)};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense_2d(const T* data, int dimx, int dimy) MUDA_NOEXCEPT
{
    return make_cdense_2d(data, make_int2(dimx, dimy));
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense_2d(T* data, int dimx, int dimy) MUDA_NOEXCEPT
{
    return make_cdense_2d(data, make_int2(dimx, dimy));
}


}  // namespace muda