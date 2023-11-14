#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cinttypes>
#include <muda/type_traits/type_modifier.h>
#include <muda/viewer/dense.h>

namespace muda
{
template <typename T>
class BufferViewBase
{
  protected:
    T*     m_data   = nullptr;
    size_t m_offset = ~0;
    size_t m_size   = ~0;

  public:
    BufferViewBase() MUDA_NOEXCEPT : m_data(nullptr), m_offset(~0), m_size(~0)
    {
    }

    BufferViewBase(T* data, size_t offset, size_t size) MUDA_NOEXCEPT
        : m_data(data),
          m_offset(offset),
          m_size(size)
    {
    }

    BufferViewBase(T* data, size_t size) MUDA_NOEXCEPT : m_data(data),
                                                         m_offset(0),
                                                         m_size(size)
    {
    }

    size_t   size() const MUDA_NOEXCEPT { return m_size; }
    const T* data() const MUDA_NOEXCEPT { return m_data + m_offset; }
    const T* origin_data() const MUDA_NOEXCEPT { return m_data; }
    size_t   offset() const MUDA_NOEXCEPT { return m_offset; }

    void copy_to(T* host) const;

    BufferViewBase subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT;
    CDense1D<T>    cviewer() const MUDA_NOEXCEPT;
};

template <typename T>
class CBufferView : public BufferViewBase<T>
{
  public:
    using BufferViewBase<T>::BufferViewBase;

    CBufferView(const BufferViewBase<T>& base)
        : BufferViewBase<T>(base)
    {
    }

    CBufferView(const T* data, size_t offset, size_t size) MUDA_NOEXCEPT
        : BufferViewBase<T>(const_cast<T*>(data), offset, size)
    {
    }

    CBufferView(const T* data, size_t size) MUDA_NOEXCEPT
        : BufferViewBase<T>(const_cast<T*>(data), size)
    {
    }

    CBufferView(CDense1D<T> viewer) MUDA_NOEXCEPT
        : BufferViewBase<T>(const_cast<T*>(viewer.data()), 0, (size_t)viewer.total_size())
    {
    }

    CBufferView subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT
    {
        return CBufferView{BufferViewBase<T>::subview(offset, size)};
    }
};

template <typename T>
class BufferView : public BufferViewBase<T>
{
  public:
    using BufferViewBase<T>::BufferViewBase;
    using BufferViewBase<T>::data;
    using BufferViewBase<T>::origin_data;

    BufferView(const BufferViewBase<T>& base)
        : BufferViewBase<T>(base)
    {
    }

    BufferView(Dense1D<T> viewer) MUDA_NOEXCEPT
        : BufferViewBase<T>(viewer.data(), 0, (size_t)viewer.total_size())
    {
    }

    operator CBufferView<T>() const MUDA_NOEXCEPT
    {
        return CBufferView<T>{*this};
    }

    T* data() MUDA_NOEXCEPT
    {
        return const_cast<T*>(BufferViewBase<T>::data());
    }

    T* origin_data() MUDA_NOEXCEPT
    {
        return const_cast<T*>(BufferViewBase<T>::origin_data());
    }

    BufferView subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT
    {
        return BufferView{BufferViewBase<T>::subview(offset, size)};
    }

    void fill(const T& v);
    void copy_from(const BufferView<T>& other);
    void copy_from(T* host);

    Dense1D<T> viewer() const MUDA_NOEXCEPT;
};

template <typename T>
struct read_only_viewer<BufferView<T>>
{
    using type = CBufferView<T>;
};

template <typename T>
struct read_write_viewer<CBufferView<T>>
{
    using type = BufferView<T>;
};

//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense(BufferView<T>& v) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense(BufferView<T>& v) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_viewer(BufferView<T>& v) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cviewer(BufferView<T>& v) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense2D(BufferView<T>& v, int dimx, int dimy) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense2D(BufferView<T>& v, int dimx, int dimy) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense2D(BufferView<T>& v, int dimy) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense2D(BufferView<T>& v, int dimy) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense2D(BufferView<T>& v, const int2& dim) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense2D(BufferView<T>& v, const int2& dim) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, int dimy, int dimz) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, int dimy, int dimz) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, const int2& dimyz) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, const int2& dimyz) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, const int3& dim) MUDA_NOEXCEPT;
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, const int3& dim) MUDA_NOEXCEPT;
}  // namespace muda

#include "details/buffer_view.inl"