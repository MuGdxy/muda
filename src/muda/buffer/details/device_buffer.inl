#include <muda/buffer/buffer_launch.h>
#include <muda/container/vector.h>

namespace muda
{
template <typename T>
DeviceBuffer<T>::DeviceBuffer(size_t n)
{
    BufferLaunch()
        .alloc(*this, n)  //
        .wait();
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer()
    : m_data(nullptr)
    , m_size(0)
    , m_capacity(0)
{
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer<T>&& other) MUDA_NOEXCEPT
    : m_data(other.m_data),
      m_size(other.m_size),
      m_capacity(other.m_capacity)
{
    other.m_data = nullptr;
    other.m_size = 0;
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const DeviceBuffer<T>& other)
{
    BufferLaunch()
        .alloc(*this, other.size())  //
        .copy(view(), other.view())  //
        .wait();
}


template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(const DeviceBuffer<T>& other)
{
    if(this == &other)
        return *this;
    BufferLaunch()
        .resize(*this, other.size())  //
        .copy(view(), other.view())   //
        .wait();
    return *this;
}

template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(BufferView<T> other)
{
    BufferLaunch()
        .resize(*this, other.size())  //
        .copy(view(), other)          //
        .wait();
    return *this;
}

template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(const std::vector<T>& other)
{
    BufferLaunch()
        .resize(*this, other.size())  //
        .copy(view(), other.data())   //
        .wait();
    return *this;
}

template <typename T>
void DeviceBuffer<T>::copy_to(T* host) const
{
    view().copy_to(host);
}

template <typename T>
void DeviceBuffer<T>::copy_to(std::vector<T>& host) const
{
    host.resize(size());
    view().copy_to(host.data());
}


template <typename T>
void DeviceBuffer<T>::resize(size_t new_size)
{
    BufferLaunch()
        .resize(*this, new_size)  //
        .wait();
}

template <typename T>
void DeviceBuffer<T>::resize(size_t new_size, const value_type& value)
{
    BufferLaunch()
        .resize(*this, new_size, value)  //
        .wait();
}

template <typename T>
void DeviceBuffer<T>::clear()
{
    BufferLaunch()
        .clear(*this)  //
        .wait();
}

template <typename T>
void DeviceBuffer<T>::shrink_to_fit()
{
    BufferLaunch()
        .shrink_to_fit(*this)  //
        .wait();
}

template <typename T>
void DeviceBuffer<T>::fill(const T& v)
{
    view().fill(v);
};

template <typename T>
Dense1D<T> DeviceBuffer<T>::viewer() MUDA_NOEXCEPT
{
    return view().viewer();
}

template <typename T>
CDense1D<T> DeviceBuffer<T>::cviewer() const MUDA_NOEXCEPT
{
    return view().cviewer();
}

template <typename T>
BufferView<T> DeviceBuffer<T>::view(size_t offset, size_t size) const MUDA_NOEXCEPT
{
    return view().subview(offset, size);
}

template <typename T>
BufferView<T> DeviceBuffer<T>::view() const MUDA_NOEXCEPT
{
    return BufferView{m_data, 0, m_size};
}

template <typename T>
DeviceBuffer<T>::~DeviceBuffer()
{
    if(m_data)
    {
        BufferLaunch()
            .free(*this)  //
            .wait();
    }
}
}  // namespace muda

namespace muda
{
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense(DeviceBuffer<T>& v) MUDA_NOEXCEPT
{
    return make_dense(v.view());
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense(const DeviceBuffer<T>& v) MUDA_NOEXCEPT
{
    return make_cdense(v.view());
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_viewer(DeviceBuffer<T>& v) MUDA_NOEXCEPT
{
    return make_viewer(v.view());
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cviewer(const DeviceBuffer<T>& v) MUDA_NOEXCEPT
{
    return make_cviewer(v.view());
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(DeviceBuffer<T>& v, int dimy) MUDA_NOEXCEPT
{
    return make_dense2D(v.view(), dimy);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense2D(const DeviceBuffer<T>& v, int dimy) MUDA_NOEXCEPT
{
    return make_cdense2D(v.view(), dimy);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(DeviceBuffer<T>& v, int dimx, int dimy) MUDA_NOEXCEPT
{
    return make_dense2D(v.view(), dimx, dimy);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense2D(const DeviceBuffer<T>& v, int dimx, int dimy) MUDA_NOEXCEPT
{
    return make_cdense2D(v.view(), dimx, dimy);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense2D(DeviceBuffer<T>& v, const int2& dim) MUDA_NOEXCEPT
{
    return make_dense2D(v.view(), dim.x, dim.y);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense2D(const DeviceBuffer<T>& v, const int2& dim) MUDA_NOEXCEPT
{
    return make_cdense2D(v.view(), dim.x, dim.y);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(DeviceBuffer<T>& v, int dimy, int dimz) MUDA_NOEXCEPT
{
    return make_dense3D(v.view(), dimy, dimz);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(const DeviceBuffer<T>& v, int dimy, int dimz) MUDA_NOEXCEPT
{
    return make_cdense3D(v.view(), dimy, dimz);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(DeviceBuffer<T>& v, const int2& dimyz) MUDA_NOEXCEPT
{
    return make_dense3D(v.view(), dimyz.x, dimyz.y);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(const DeviceBuffer<T>& v, const int2& dimyz) MUDA_NOEXCEPT
{
    return make_cdense3D(v.view(), dimyz.x, dimyz.y);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(DeviceBuffer<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return make_dense3D(v.view(), dimx, dimy, dimz);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(const DeviceBuffer<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
{
    return make_cdense3D(v.view(), dimx, dimy, dimz);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_dense3D(DeviceBuffer<T>& v, const int3& dim) MUDA_NOEXCEPT
{
    return make_dense3D(v.view(), dim.x, dim.y, dim.z);
}
template <typename T>
MUDA_INLINE MUDA_HOST auto make_cdense3D(const DeviceBuffer<T>& v, const int3& dim) MUDA_NOEXCEPT
{
    return make_cdense3D(v.view(), dim.x, dim.y, dim.z);
}
}