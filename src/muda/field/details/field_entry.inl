#include <muda/field/sub_field.h>

namespace muda
{
MUDA_INLINE FieldEntryViewerBase FieldEntryBase::viewer()
{
    MUDA_ASSERT(m_field.data_buffer() != nullptr, "Resize the field before you use it!");
    return FieldEntryViewerBase{m_field.data_buffer(), m_info, m_name_ptr};
}

template <typename T, FieldEntryLayout Layout, int M, int N>
FieldEntryViewer<T, Layout, M, N> FieldEntry<T, Layout, M, N>::viewer()
{
    MUDA_ASSERT(m_field.data_buffer() != nullptr, "Resize the field before you use it!");
    return FieldEntryViewer<T, Layout, M, N>{m_field.data_buffer(), m_info, m_name_ptr};
}
template <typename T, FieldEntryLayout Layout, int M, int N>
CFieldEntryViewer<T, Layout, M, N> muda::FieldEntry<T, Layout, M, N>::cviewer() const
{
    MUDA_ASSERT(m_field.data_buffer() != nullptr, "Resize the field before you use it!");
    return CFieldEntryViewer<T, Layout, M, N>{m_field.data_buffer(), m_info, m_name_ptr};
}
}  // namespace muda