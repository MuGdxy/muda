#include <muda/ext/field/sub_field.h>

namespace muda
{
//using T                  = float;
//constexpr auto Layout    = FieldEntryLayout::RuntimeLayout;
//constexpr auto SrcLayout = FieldEntryLayout::RuntimeLayout;
//constexpr auto M         = 3;
//constexpr auto N         = 3;
template <typename T, FieldEntryLayout Layout, int M, int N>
void FieldEntry<T, Layout, M, N>::copy_to(DeviceBuffer<ElementType>& dst) const
{
    dst.resize(count());
    view().copy_to(dst);
}

template <typename T, FieldEntryLayout Layout, int M, int N>
void FieldEntry<T, Layout, M, N>::copy_to(std::vector<ElementType>& dst) const
{
    dst.resize(count());
    m_workpace.resize(count());
    FieldEntryLaunch().copy(m_workpace.view(), view());
    BufferLaunch().copy(dst.data(), std::as_const(m_workpace).view()).wait();
}

template <typename T, FieldEntryLayout Layout, int M, int N>
void FieldEntry<T, Layout, M, N>::copy_from(const DeviceBuffer<ElementType>& src)
{
    MUDA_ASSERT(src.size() == count(),
                "FieldEntry: size mismatch, src.size()=%d, this count()=%d, field entry can't resize itself when copying!",
                src.size(),
                count());
    view().copy_from(src.view());
}

template <typename T, FieldEntryLayout Layout, int M, int N>
void FieldEntry<T, Layout, M, N>::copy_from(const std::vector<ElementType>& src)
{
    MUDA_ASSERT(src.size() == count(),
                "FieldEntry: size mismatch, src.size()=%d, this count()=%d, field entry can't resize itself when copying!",
                src.size(),
                count());
    m_workpace.resize(count());
    BufferLaunch().copy(m_workpace.view(), src.data()).wait();
    FieldEntryLaunch().copy(view(), std::as_const(m_workpace).view()).wait();
}

template <typename T, FieldEntryLayout Layout, int M, int N>
void FieldEntry<T, Layout, M, N>::fill(const ElementType& value)
{
    view().fill(value);
}

template <typename T, FieldEntryLayout Layout, int M, int N>
template <FieldEntryLayout SrcLayout>
void FieldEntry<T, Layout, M, N>::copy_from(const FieldEntry<T, SrcLayout, M, N>& src)
{
    MUDA_ASSERT(src.count() == count(),
                "FieldEntry: size mismatch, src.count()=%d, this count()=%d, field entry can't resize itself when copying!",
                src.count(),
                count());
    view().copy_from(src.view());
}
}  // namespace muda