#include <muda/buffer/buffer_launch.h>

namespace muda
{
template <typename T>
void VarView<T>::copy_from(const T* val)
{
    BufferLaunch()
        .copy(*this, val)  //
        .wait();
}

template <typename T>
void CVarView<T>::copy_to(T* val) const
{
    BufferLaunch()
        .copy(val, *this)  //
        .wait();
}

template <typename T>
void VarView<T>::copy_from(CVarView<T> val)
{
    BufferLaunch()
        .copy(*this, val)  //
        .wait();
}

template <typename T>
void VarView<T>::fill(const T& val)
{
    BufferLaunch()
        .fill(*this, val)  //
        .wait();
}

template <typename T>
Dense<T> VarView<T>::viewer() MUDA_NOEXCEPT
{
    return Dense<T>{m_data};
}

template <typename T>
CDense<T> VarViewBase<T>::cviewer() const MUDA_NOEXCEPT
{
    return CDense<T>{m_data};
}
}  // namespace muda