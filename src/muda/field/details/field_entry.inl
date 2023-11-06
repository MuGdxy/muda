namespace muda
{
MUDA_INLINE FieldEntryViewerBase FieldEntryBase::viewer()
{
    return FieldEntryViewerBase{m_field.m_data_buffer.data(), m_info, m_name_ptr};
}

template <typename T, FieldEntryLayout Layout, int M, int N>
FieldEntryViewer<T, Layout, M, N> FieldEntry<T, Layout, M, N>::viewer()
{
    return FieldEntryViewer<T, Layout, M, N>{m_field.m_data_buffer.data(), m_info, m_name_ptr};
}
template <typename T, FieldEntryLayout Layout, int M, int N>
CFieldEntryViewer<T, Layout, M, N> muda::FieldEntry<T, Layout, M, N>::cviewer() const
{
    return CFieldEntryViewer<T, Layout, M, N>{m_field.m_data_buffer.data(), m_info, m_name_ptr};
}
}  // namespace muda