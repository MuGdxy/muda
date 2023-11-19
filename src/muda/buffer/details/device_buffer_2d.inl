#include <muda/buffer/buffer_launch.h>

namespace muda
{
template <typename T>
DeviceBuffer2D<T>::DeviceBuffer2D(const Extent2D& n)
{
    BufferLaunch()
        .resize(*this, n)  //
        .wait();
}

template <typename T>
DeviceBuffer2D<T>::DeviceBuffer2D()
    : m_data(nullptr)
    , m_pitch_bytes(0)
    , m_extent(Extent2D::Zero())
    , m_capacity(Extent2D::Zero())
{
}

template <typename T>
DeviceBuffer2D<T>::DeviceBuffer2D(const DeviceBuffer2D<T>& other)
{
    BufferLaunch()
        .resize(*this, other.extent())  //
        .copy(view(), other.view())     //
        .wait();
}

template <typename T>
DeviceBuffer2D<T>::DeviceBuffer2D(DeviceBuffer2D<T>&& other) MUDA_NOEXCEPT
    : m_data(other.m_data),
      m_pitch_bytes(other.m_pitch_bytes),
      m_extent(other.m_extent),
      m_capacity(other.m_capacity)
{
    other.m_data        = nullptr;
    other.m_pitch_bytes = 0;
    other.m_extent      = Extent2D::Zero();
    other.m_capacity    = Extent2D::Zero();
}

template <typename T>
DeviceBuffer2D<T>& DeviceBuffer2D<T>::operator=(const DeviceBuffer2D<T>& other)
{
    if(this == &other)
        return *this;

    BufferLaunch()
        .resize(*this, other.extent())  //
        .copy(view(), other.view())     //
        .wait();

    return *this;
}

template <typename T>
DeviceBuffer2D<T>& DeviceBuffer2D<T>::operator=(DeviceBuffer2D<T>&& other)
{
    if(this == &other)
        return *this;

    if(m_data)
        BufferLaunch().free(*this).wait();

    m_data        = other.m_data;
    m_pitch_bytes = other.m_pitch_bytes;
    m_extent      = other.m_extent;
    m_capacity    = other.m_capacity;

    other.m_data        = nullptr;
    other.m_pitch_bytes = 0;
    other.m_extent      = Extent2D::Zero();
    other.m_capacity    = Extent2D::Zero();

    return *this;
}

template <typename T>
DeviceBuffer2D<T>::DeviceBuffer2D(CBuffer2DView<T> other)
{
    BufferLaunch()
        .alloc(*this, other.extent())  //
        .copy(view(), other)           //
        .wait();
}

template <typename T>
DeviceBuffer2D<T>& DeviceBuffer2D<T>::operator=(CBuffer2DView<T> other)
{
    BufferLaunch()
        .resize(*this, other.extent())  //
        .copy(view(), other)            //
        .wait();

    return *this;
}


template <typename T>
void DeviceBuffer2D<T>::copy_to(std::vector<T>& host) const
{
    host.resize(total_size());
    view().copy_to(host.data());
}


template <typename T>
void DeviceBuffer2D<T>::copy_from(const std::vector<T>& host)
{
    MUDA_ASSERT(host.size() == total_size(),
                "Need eqaul total size, host_size=%d, total_size=%d",
                host.size(),
                total_size());

    view().copy_from(host.data());
}

template <typename T>
void DeviceBuffer2D<T>::resize(Extent2D new_extent)
{
    BufferLaunch()
        .resize(*this, new_extent)  //
        .wait();
}

template <typename T>
void DeviceBuffer2D<T>::resize(Extent2D new_extent, const T& value)
{
    BufferLaunch()
        .resize(*this, new_extent, value)  //
        .wait();
}

template <typename T>
void DeviceBuffer2D<T>::reserve(Extent2D new_capacity)
{
    BufferLaunch()
        .reserve(*this, new_capacity)  //
        .wait();
}

template <typename T>
void DeviceBuffer2D<T>::clear()
{
    BufferLaunch().clear(*this).wait();
}

template <typename T>
void DeviceBuffer2D<T>::shrink_to_fit()
{
    BufferLaunch().shrink_to_fit(*this).wait();
}

template <typename T>
void DeviceBuffer2D<T>::fill(const T& v)
{
    BufferLaunch().fill(view(), v).wait();
}

template <typename T>
DeviceBuffer2D<T>::~DeviceBuffer2D()
{
    if(m_data)
        BufferLaunch().free(*this).wait();
}
}  // namespace muda