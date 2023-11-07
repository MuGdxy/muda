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
    return i & (layout_info().innermost_array_size() - 1);
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::aosoa_struct_begin(int i) const
{
    auto outer_index = i / layout_info().innermost_array_size();
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
    auto innermost_array_size = layout_info().innermost_array_size();
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
    auto innermost_array_size = layout_info().innermost_array_size();
    auto struct_begin         = aosoa_struct_begin(i);
    auto inner_index          = aosoa_inner_index(i);
    // column major
    auto j = col_index * shape().x + row_index;
    return struct_begin
           + elem_byte_size() * (innermost_array_size * j + aosoa_inner_index(i));
}


MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::soa_elem_addr(int i) const
{
    MUDA_KERNEL_ASSERT(i < count(),
                       "FieldEntry[%s:%s]: count indexing out of range, size=%d, index=%d",
                       kernel_name(),
                       name(),
                       count(),
                       i);

    return m_buffer + m_info.offset_in_struct + m_info.elem_byte_size * i;
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::soa_elem_addr(int i, int comp_j) const
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
    auto offset = m_info.elem_count_based_stride * comp_j + m_info.elem_byte_size * i;
    return m_buffer + m_info.offset_in_struct + offset;
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::soa_elem_addr(int i,
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
    // column major
    auto j = col_index * shape().x + row_index;
    auto offset = m_info.elem_count_based_stride * j + m_info.elem_byte_size * i;
    return m_buffer + m_info.offset_in_struct + offset;
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::aos_struct_begin(int i) const
{
    return m_buffer + m_info.struct_stride * i;
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::aos_elem_addr(int i) const
{
    MUDA_KERNEL_ASSERT(i < count(),
                       "FieldEntry[%s:%s]: count indexing out of range, size=%d, index=%d",
                       kernel_name(),
                       name(),
                       count(),
                       i);

    return aos_struct_begin(i) + m_info.offset_in_struct;
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::aos_elem_addr(int i, int comp_j) const
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

    return aos_struct_begin(i) + m_info.offset_in_struct + m_info.elem_byte_size * comp_j;
}

MUDA_INLINE MUDA_GENERIC std::byte* FieldEntryViewerBase::aos_elem_addr(int i,
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
    // column major
    auto j = col_index * shape().x + row_index;

    return aos_struct_begin(i) + m_info.offset_in_struct + m_info.elem_byte_size * j;
}

}  // namespace muda