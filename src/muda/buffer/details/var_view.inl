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
}  // namespace muda