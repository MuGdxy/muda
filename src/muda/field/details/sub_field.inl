#include <muda/field/field.h>
#include <muda/field/field_entry.h>
#include <muda/field/field_builder.h>

namespace muda
{
MUDA_INLINE SubField::SubField(Field& field, std::string_view name)
    : m_field(field)
    , m_name(name)
{
}

MUDA_INLINE SubField::~SubField()
{
    for(auto& entry : m_entries)
        delete entry;
}
template <FieldEntryLayout Layout>
MUDA_INLINE FieldBuilder<Layout> SubField::builder(FieldEntryLayoutInfo layout)
{
    m_layout = layout;
    return FieldBuilder<Layout>{*this, layout};
}

MUDA_INLINE FieldBuilder<FieldEntryLayout::RuntimeLayout> SubField::builder(FieldEntryLayoutInfo layout)
{
    m_layout = layout;
    return FieldBuilder<FieldEntryLayout::RuntimeLayout>{*this, layout};
}
template <typename T, FieldEntryLayout Layout, int M, int N>
auto SubField::create_entry(std::string_view     name,
                            FieldEntryLayoutInfo layout,
                            FieldEntryType       type,
                            uint2 shape) -> FieldEntry<T, Layout, M, N>&
{
    auto ptr = new FieldEntry<T, Layout, M, N>(*this, layout, type, shape, name);
    m_name_to_index[std::string{name}] = m_entries.size();
    m_entries.emplace_back(ptr);
    return *ptr;
}

MUDA_INLINE auto SubField::find_entry(std::string_view name) const -> FieldEntryBase*
{
    auto it = m_name_to_index.find(std::string{name});
    if(it == m_name_to_index.end())
    {
        return nullptr;
    }
    return m_entries[it->second];
}

MUDA_INLINE FieldBuilder<FieldEntryLayout::AoSoA> SubField::AoSoA(FieldEntryLayoutInfo layout)
{
    return builder<FieldEntryLayout::AoSoA>(layout);
}

MUDA_INLINE FieldBuilder<FieldEntryLayout::SoA> SubField::SoA()
{
    return builder<FieldEntryLayout::SoA>(FieldEntryLayoutInfo{FieldEntryLayout::SoA, 0});
}

MUDA_INLINE FieldBuilder<FieldEntryLayout::AoS> SubField::AoS()
{
    return builder<FieldEntryLayout::AoS>(FieldEntryLayoutInfo{FieldEntryLayout::AoS, 0});
}

template <typename F>
void SubField::resize_data_buffer(F&& func, size_t size)
{
    if(m_data_buffer == nullptr)
    {
        Memory().alloc(&m_data_buffer, size).set(m_data_buffer, size, 0).wait();
        m_data_buffer_size = size;
    }
    else
    {
        auto old_ptr       = m_data_buffer;
        auto old_size      = m_data_buffer_size;
        m_data_buffer_size = size;
        Memory().alloc(&m_data_buffer, size);
        func(old_ptr, old_size, m_data_buffer, size);
        Memory().free(old_ptr).wait();
    }
}

MUDA_INLINE void SubField::copy_resize_data_buffer(size_t size)
{
    resize_data_buffer(
        [](std::byte* old_ptr, size_t old_size, std::byte* new_ptr, size_t new_size)
        {
            Memory()
                .set(new_ptr + old_size, new_size - old_size, 0)  // set the new memory to 0
                .transfer(new_ptr, old_ptr, old_size);  // copy the old memory to the new memory
        },
        size);
}

MUDA_INLINE void SubField::upload_entries() const
{
    //m_entries_buffer.resize(m_entries.size());
    //std::vector<FieldEntryViewerBase> entries(m_entries.size());
    //std::transform(m_entries.begin(),
    //               m_entries.end(),
    //               entries.begin(),
    //               [](auto entry) { return entry->viewer(); });
    //// host to device
    //m_entries_buffer.copy_from(entries);
}

MUDA_INLINE uint32_t SubField::div_round_up(uint32_t x, uint32_t n)
{
    MUDA_ASSERT((n & (n - 1)) == 0, "n is not power of 2");
    return (x + n - 1) & ~(n - 1);
}

MUDA_INLINE uint32_t SubField::align(uint32_t offset, uint32_t size, uint32_t min_alignment, uint32_t max_alignment)
{
    auto alignment = std::clamp(size, min_alignment, max_alignment);
    return div_round_up(offset, alignment);
}

MUDA_INLINE void SubField::build(const FieldBuildOptions& options)
{
    MUDA_ASSERT(!m_is_built, "Field is already built!");
    switch(m_layout.layout())
    {
        case FieldEntryLayout::AoSoA:
            build_aosoa(options);
            break;
        case FieldEntryLayout::SoA:
            MUDA_ERROR_WITH_LOCATION("SoA is not supported yet");
            break;
        case FieldEntryLayout::AoS:
            build_aos(options);
            break;
        default:
            MUDA_ERROR_WITH_LOCATION("Unknown layout");
            break;
    }
    m_is_built = true;
}

MUDA_INLINE size_t SubField::struct_stride() const
{
    MUDA_ASSERT(m_is_built, "Build the field before getting struct_stride");
    return m_struct_stride;
}

MUDA_INLINE void SubField::build_aosoa(const FieldBuildOptions& options)
{
    auto min_alignment = options.min_alignment;
    auto max_alignment = options.max_alignment;
    // eg: innermost array size = 4
    // a "Struct" is something like the following, where M/V/S are 3 different entries, has type of matrix/vector/scalar
    //tex:
    // $$
    // \begin{bmatrix}
    // M_{11} & M_{11} & M_{11} & M_{11}\\
    // M_{21} & M_{21} & M_{21} & M_{21}\\
    // M_{12} & M_{12} & M_{12} & M_{12}\\
    // M_{22} & M_{22} & M_{22} & M_{22}\\
    // V_x & V_x & V_x & V_x\\
    // V_y & V_y & V_y & V_y\\
    // V_z & V_z & V_z & V_z\\
    // S   & S   & S   & S \\
    // \end{bmatrix}
    // $$
    uint32_t struct_stride = 0;  // the stride of the "Struct"
    for(auto e : m_entries)  // in an entry, the elem type is the same (e.g. float/int/double...)
    {
        // elem type = float/double/int ... or User Type
        auto elem_byte_size = e->elem_byte_size();
        // innermost array size: most of time the size = 32 (warp size)
        auto inner_array_size = e->layout().innermost_array_size();
        // total elem count in innermost array:
        // scalar=1 vector3 = 3, vector4 = 4, matrix3x3 = 9, matrix4x4 = 16, and so on

        struct_stride = align(struct_stride, elem_byte_size, min_alignment, max_alignment);
        // now struct_stride is the offset of the entry in the "Struct"
        e->m_info.offset_in_struct = struct_stride;

        auto total_elem_count_in_innermost_array = e->shape().x * e->shape().y * inner_array_size;
        struct_stride += elem_byte_size * total_elem_count_in_innermost_array;
    }
    // the final stride of the "Struct" >= struct size
    m_struct_stride = align(struct_stride, struct_stride, min_alignment, max_alignment);

    for(auto e : m_entries)
    {
        e->m_info.struct_stride = m_struct_stride;
        e->m_name_ptr           = m_field.m_string_cache[e->m_name];
    }
}

MUDA_INLINE void SubField::build_soa(const FieldBuildOptions& options)
{
    // the array size is unknown so we can't build the field now, 
    // we will build the field when resize() is called
    for(auto e : m_entries)
        e->m_name_ptr           = m_field.m_string_cache[e->m_name];
    
}

MUDA_INLINE void SubField::build_aos(const FieldBuildOptions& options)
{
    auto min_alignment = options.min_alignment;
    auto max_alignment = options.max_alignment;
    // a "Struct" is something like the following, where M/V/S are 3 different entries, has type of matrix/vector/scalar
    //tex:
    // $$
    // \begin{bmatrix}
    // M_{11} & M_{21} & M_{12} & M_{22} & V_x & V_y & V_z & S
    // \end{bmatrix}
    // $$
    uint32_t struct_stride = 0;  // the stride of the "Struct"
    for(auto e : m_entries)  // in an entry, the elem type is the same (e.g. float/int/double...)
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

    for(auto e : m_entries)
    {
        e->m_info.struct_stride = m_struct_stride;
        e->m_name_ptr           = m_field.m_string_cache[e->m_name];
    }
}

MUDA_INLINE void SubField::resize_build_soa(const FieldBuildOptions& options, size_t num_elements)
{
    //auto min_alignment = options.min_alignment;
    //auto max_alignment = options.max_alignment;
    //// e.g. array size = 4
    //// a "Struct" is something like the following, where M/V/S are 3 different entries, has type of matrix/vector/scalar
    ////tex:
    //// $$
    //// \begin{bmatrix}
    //// M_{11} & M_{11} & M_{11} & M_{11}\\
    //// M_{21} & M_{21} & M_{21} & M_{21}\\
    //// M_{12} & M_{12} & M_{12} & M_{12}\\
    //// M_{22} & M_{22} & M_{22} & M_{22}\\
    //// V_x & V_x & V_x & V_x\\
    //// V_y & V_y & V_y & V_y\\
    //// V_z & V_z & V_z & V_z\\
    //// S   & S   & S   & S \\
    //// \end{bmatrix}
    //// $$
    //uint32_t struct_stride = 0;  // the stride of the "Struct"=> SoA total size
    //for(auto e : m_entries)  // in an entry, the elem type is the same (e.g. float/int/double...)
    //{
    //    // elem type = float/double/int ... or User Type
    //    auto elem_byte_size = e->elem_byte_size();
    //    // total elem count in innermost array:
    //    // scalar=1 vector3 = 3, vector4 = 4, matrix3x3 = 9, matrix4x4 = 16, and so on
    //    auto elem_count = e->shape().x * e->shape().y;
    //    struct_stride = align(struct_stride, elem_byte_size, min_alignment, max_alignment);

    //    // now struct_stride is the offset of the entry in the "Struct"
    //    e->m_info.offset_in_struct = struct_stride;

    //    struct_stride += elem_byte_size * inner_array_size * total_elem_count_in_innermost_array;
    //}
    //// the final stride of the "Struct" >= struct size
    //m_struct_stride = align(struct_stride, struct_stride, min_alignment, max_alignment);
}

MUDA_INLINE void SubField::resize(size_t num_elements)
{
    switch(m_layout.layout())
    {
        case FieldEntryLayout::AoSoA:
            resize_aosoa(num_elements);
            break;
        case FieldEntryLayout::SoA:
            MUDA_ERROR_WITH_LOCATION("SoA is not supported yet");
            break;
        case FieldEntryLayout::AoS:
            resize_aos(num_elements);
            break;
        default:
            MUDA_ERROR_WITH_LOCATION("Unknow Layout");
            break;
    }

    m_num_elements = num_elements;
}

MUDA_INLINE void SubField::resize_aosoa(size_t num_elements)
{
    size_t outer_size = div_round_up(num_elements, m_layout.innermost_array_size());
    copy_resize_data_buffer(outer_size * m_struct_stride);
}

MUDA_INLINE void SubField::resize_soa(size_t num_elements) {}

MUDA_INLINE void SubField::resize_aos(size_t num_elements)
{
    copy_resize_data_buffer(num_elements);
}

}  // namespace muda