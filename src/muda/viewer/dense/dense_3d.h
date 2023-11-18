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
    int3 m_offset;
    int3 m_dim;
    int  m_pitch_bytes;
    int  m_pitch_bytes_area;

  public:
    using value_type = T;

    MUDA_GENERIC Dense3DBase() MUDA_NOEXCEPT : m_data(nullptr){};

    MUDA_GENERIC Dense3DBase(T* p, const int3& offset, const int3& dim, int pitch_bytes, int pitch_bytes_area) MUDA_NOEXCEPT
        : m_data(p),
          m_offset(offset),
          m_dim(dim),
          m_pitch_bytes(pitch_bytes),
          m_pitch_bytes_area(pitch_bytes_area)
    {
    }

    MUDA_GENERIC const T& operator()(const int3& xyz) const MUDA_NOEXCEPT
    {
        return operator()(xyz.x, xyz.y, xyz.z);
    }

    MUDA_GENERIC const T& operator()(int x, int y, int z) const MUDA_NOEXCEPT
    {
        check();
        check_range(x, y, z);
        auto depth_begin = reinterpret_cast<std::byte*>(m_data) + x * m_pitch_bytes_area;
        auto height_begin = depth_begin + y * m_pitch_bytes;
        return *(reinterpret_cast<T*>(height_begin) + z);
    }

    MUDA_GENERIC const T& flatten(int i) const MUDA_NOEXCEPT
    {
        auto area       = m_dim.y * m_dim.z;
        auto x          = i / area;
        auto i_in_area  = i % area;
        auto y          = i_in_area / m_dim.z;
        auto i_in_width = i_in_area % m_dim.z;
        auto z          = i_in_width;
        return operator()(x, y, z);
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC auto dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC int area() const MUDA_NOEXCEPT { return m_dim.y * m_dim.z; }

    MUDA_GENERIC int volume() const MUDA_NOEXCEPT { return total_size(); }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT
    {
        return m_dim.x * area();
    }

    MUDA_GENERIC int pitch_bytes() const MUDA_NOEXCEPT { return m_pitch_bytes; }

    MUDA_GENERIC int pitch_bytes_area() const MUDA_NOEXCEPT
    {
        return m_pitch_bytes_area;
    }

    MUDA_GENERIC int total_bytes() const MUDA_NOEXCEPT
    {
        return m_pitch_bytes_area * m_dim.x;
    }

  protected:
    MUDA_INLINE MUDA_GENERIC void check_range(int x, int y, int z) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
        {
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
        }
    }

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
    using Base = Dense3DBase<T>;

  public:
    using Base::Base;
    using Base::operator();

    MUDA_GENERIC CDense3D(const Base& base) MUDA_NOEXCEPT : Base(base){};

    MUDA_GENERIC CDense3D(const T* p, const int3& offset, const int3& dim, int pitch_bytes, int pitch_bytes_area) MUDA_NOEXCEPT
        : Base(const_cast<T*>(p), offset, dim, pitch_bytes, pitch_bytes_area)
    {
    }
};

template <typename T>
class Dense3D : public Dense3DBase<T>
{
    MUDA_VIEWER_COMMON_NAME(Dense3D);
    using Base = Dense3DBase<T>;

  public:
    using Base::Base;
    using Base::operator();
    using Base::data;
    using Base::flatten;

    MUDA_GENERIC Dense3D(const Base& base) MUDA_NOEXCEPT : Base(base) {}

    MUDA_GENERIC operator CDense3D<T>() const MUDA_NOEXCEPT
    {
        return CDense3D<T>{*this};
    }

    MUDA_GENERIC T& operator()(const int3& xyz) MUDA_NOEXCEPT
    {
        return const_cast<T&>(Base::operator()(xyz));
    }

    MUDA_GENERIC T& operator()(int x, int y, int z) MUDA_NOEXCEPT
    {
        return const_cast<T&>(Base::operator()(x, y, z));
    }

    MUDA_GENERIC T& flatten(int i) MUDA_NOEXCEPT
    {
        return const_cast<T&>(Base::flatten(i));
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT
    {
        return const_cast<T&>(Base::data());
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

// make functions
template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense_3d(const T* data, const int3& dim) MUDA_NOEXCEPT
{
    auto pitch_bytes = dim.z * sizeof(T);
    return CDense3D<T>{data,
                       make_int3(0, 0, 0),
                       dim,
                       static_cast<int>(pitch_bytes),
                       static_cast<int>(dim.y * pitch_bytes)};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense_3d(T* data, const int3& dim) MUDA_NOEXCEPT
{
    auto pitch_bytes = dim.z * sizeof(T);
    return Dense3D<T>{data,
                      make_int3(0, 0, 0),
                      dim,
                      static_cast<int>(pitch_bytes),
                      static_cast<int>(dim.y * pitch_bytes)};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense_3d(const T* data, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return make_cdense_3d(data, make_int3(dimx, dimy, dimz));
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense_3d(T* data, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return make_dense_3d(data, make_int3(dimx, dimy, dimz));
}
}  // namespace muda