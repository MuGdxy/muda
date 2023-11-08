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
class BufferView
{

    template <typename T>
    friend class DeviceBuffer;
    T*     m_data   = nullptr;
    size_t m_offset = ~0;
    size_t m_size   = ~0;

  public:
    BufferView() MUDA_NOEXCEPT : m_data(nullptr), m_offset(~0), m_size(~0) {}

    BufferView(T* data, size_t offset, size_t size) MUDA_NOEXCEPT : m_data(data),
                                                                    m_offset(offset),
                                                                    m_size(size)
    {
    }

    BufferView(T* data, size_t size) MUDA_NOEXCEPT : m_data(data), m_offset(0), m_size(size)
    {
    }

    BufferView(const Dense1D<T>& viewer) MUDA_NOEXCEPT
        : m_data(viewer.data()),
          m_offset(0),
          m_size((size_t)viewer.total_size())
    {
    }

    size_t   size() const MUDA_NOEXCEPT { return m_size; }
    T*       data() MUDA_NOEXCEPT { return m_data + m_offset; }
    const T* data() const MUDA_NOEXCEPT { return m_data + m_offset; }
    T*       origin_data() MUDA_NOEXCEPT { return m_data; }
    const T* origin_data() const MUDA_NOEXCEPT { return m_data; }
    size_t   offset() const MUDA_NOEXCEPT { return m_offset; }

    BufferView subview(size_t offset, size_t size = ~0) const MUDA_NOEXCEPT;

    void fill(const T& v);
    void copy_from(const BufferView<T>& other);
    void copy_from(T* host);
    void copy_to(T* host) const;

    Dense1D<T>  viewer() MUDA_NOEXCEPT;
    CDense1D<T> cviewer() const MUDA_NOEXCEPT;
};

template <typename T>
struct read_only_viewer<BufferView<T>>
{
    using type = const BufferView<T>;
};

template <typename T>
struct read_write_viewer<const BufferView<T>>
{
    using type = BufferView<T>;
};

template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense(BufferView<T>& v) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense(BufferView<T>& v) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_viewer(BufferView<T>& v) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cviewer(BufferView<T>& v) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(BufferView<T>& v, int dimx, int dimy) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense2D(BufferView<T>& v, int dimx, int dimy) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(BufferView<T>& v, int dimy) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense2D(BufferView<T>& v, int dimy) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(BufferView<T>& v, const int2& dim) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense2D(BufferView<T>& v, const int2& dim) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, int dimy, int dimz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, int dimy, int dimz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, const int2& dimyz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, const int2& dimyz) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, const int3& dim) MUDA_NOEXCEPT;
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, const int3& dim) MUDA_NOEXCEPT;
}  // namespace muda

#include "details/buffer_view.inl"