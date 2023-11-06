namespace muda
{
template <typename T>
MUDA_INLINE MUDA_GENERIC T& FieldEntryViewerBase::cast(std::byte* data)
{
    return *(reinterpret_cast<T*>(data));
}

template <typename T>
MUDA_INLINE MUDA_GENERIC const T& FieldEntryViewerBase::cast(const std::byte* data) const
{
    return *(reinterpret_cast<const T*>(data));
}

MUDA_INLINE MUDA_GENERIC uint32_t FieldEntryViewerBase::aosoa_inner_index(int i) const
{
    return i & (m_layout.innermost_array_size() - 1);
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::aosoa_struct_begin(int i) const
{
    auto outer_index = i / m_layout.innermost_array_size();
    return m_buffer + outer_index * struct_stride() + m_info.offset_in_struct;
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::aosoa_elem_addr(int i) const
{
    MUDA_KERNEL_ASSERT(i < count(),
                       "FieldEntry[%s:%s]: count indexing out of range, size=%d, index=%d",
                       kernel_name(),
                       name(),
                       count(),
                       i);

    return aosoa_struct_begin(i) + elem_byte_size() * aosoa_inner_index(i);
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::aosoa_elem_addr(int i, int comp_j) const
{
    MUDA_KERNEL_ASSERT(i < count(),
                       "FieldEntry[%s:%s]: count indexing out of range, size=%d, index=%d",
                       kernel_name(),
                       name(),
                       count(),
                       i);

    MUDA_KERNEL_ASSERT(comp_j < shape().x,
                       "FieldEntry[%s:%s]: vector component indexing out of range, shape=(%d, %d), index=%d",
                       kernel_name(),
                       name(),
                       shape().x,
                       shape().y,
                       comp_j);
    auto innermost_array_size = m_layout.innermost_array_size();
    auto struct_begin         = aosoa_struct_begin(i);
    auto inner_index          = aosoa_inner_index(i);
    return struct_begin
           + elem_byte_size() * (innermost_array_size * comp_j + aosoa_inner_index(i));
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::aosoa_elem_addr(int i,
                                                                          int row_index,
                                                                          int col_index) const
{
    MUDA_KERNEL_ASSERT(i < count(),
                       "FieldEntry[%s:%s]: count indexing out of range, size=%d, index=%d",
                       kernel_name(),
                       name(),
                       count(),
                       i);

    MUDA_KERNEL_ASSERT(row_index < shape().x && col_index < shape().y,
                       "FieldEntry[%s:%s]: vector component indexing out of range, shape=(%d,%d), index=(%d,%d)",
                       kernel_name(),
                       name(),
                       shape().x,
                       shape().y,
                       row_index,
                       col_index);
    auto innermost_array_size = m_layout.innermost_array_size();
    auto struct_begin         = aosoa_struct_begin(i);
    auto inner_index          = aosoa_inner_index(i);
    // column major
    auto j = col_index * shape().x + row_index;
    return struct_begin
           + elem_byte_size() * (innermost_array_size * j + aosoa_inner_index(i));
}
}  // namespace muda