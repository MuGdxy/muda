namespace muda
{
template <typename T>
MUDA_INLINE MUDA_GENERIC T& FieldEntryViewerBase::cast(void* data)
{
    return *(reinterpret_cast<T*>(data));
}

template <typename T>
MUDA_INLINE MUDA_GENERIC const T& FieldEntryViewerBase::cast(const void* data) const
{
    return *(reinterpret_cast<const T*>(data));
}

MUDA_INLINE MUDA_GENERIC void* FieldEntryViewerBase::aosoa_elem_addr(int i, int comp_j) const
{
    MUDA_KERNEL_ASSERT(i < count(),
                       "FieldEntry[%s:%s]: count indexing out of range, size=%d (index=%d)",
                       kernel_name(),
                       name(),
                       count(),
                       i);

    MUDA_KERNEL_ASSERT(comp_j < shape().x * shape().y,
                       "FieldEntry[%s:%s]: component indexing out of range, shpae=(%d, %d) size=%d index=%d",
                       kernel_name(),
                       name(),
                       shape().x,
                       shape().y,
                       shape().x * shape().y,
                       comp_j);
    auto innermost_array_size = m_layout.innermost_array_size();
    auto outer_index          = i / innermost_array_size;
    // calculate inner index
    auto inner_index  = i & (innermost_array_size - 1);
    auto struct_begin = m_buffer + outer_index * struct_stride() + m_info.begin;
    return struct_begin + elem_byte_size() * (innermost_array_size * comp_j + inner_index);
}
}  // namespace muda