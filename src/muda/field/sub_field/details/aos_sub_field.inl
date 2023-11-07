namespace muda
{
MUDA_INLINE void SubFieldImpl<FieldEntryLayout::AoS>::build()
{
    auto min_alignment = build_options().min_alignment;
    auto max_alignment = build_options().max_alignment;
    // a "Struct" is something like the following, where M/V/S are 3 different entries, has type of matrix/vector/scalar
    //tex:
    // $$
    // \begin{bmatrix}
    // M_{11} & M_{21} & M_{12} & M_{22} & V_x & V_y & V_z & S
    // \end{bmatrix}
    // $$
    uint32_t struct_stride = 0;  // the stride of the "Struct"
    for(auto& e : m_entries)  // in an entry, the elem type is the same (e.g. float/int/double...)
    {
        // elem type = float/double/int ... or User Type
        auto elem_byte_size = e->elem_byte_size();
        // e.g. scalar=1 vector3 = 3, vector4 = 4, matrix3x3 = 9, matrix4x4 = 16, and so on
        auto total_elem_count_in_a_struct_member = e->shape().x * e->shape().y;
        struct_stride = align(struct_stride, elem_byte_size, min_alignment, max_alignment);
        // now struct_stride is the offset of the entry in the "Struct"
        e->m_info.offset_in_struct = struct_stride;
        struct_stride += elem_byte_size * total_elem_count_in_a_struct_member;
    }
    // the final stride of the "Struct" >= struct size
    m_struct_stride = align(struct_stride, struct_stride, min_alignment, max_alignment);

    for(auto& e : m_entries)
    {
        e->m_info.struct_stride = m_struct_stride;
        e->m_name_ptr           = m_field.m_string_cache[e->m_name];
    }
}

MUDA_INLINE void SubFieldImpl<FieldEntryLayout::AoS>::resize(size_t num_elements)
{
    copy_resize_data_buffer(m_struct_stride * num_elements);
    for(auto& e : m_entries)
    {
        e->m_info.elem_count = num_elements;
    }
}
}  // namespace muda
