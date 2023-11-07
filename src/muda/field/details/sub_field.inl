#include <muda/field/field.h>
#include <muda/field/field_entry.h>
#include <muda/field/field_builder.h>
#include <muda/cuda/cooperative_groups.h>
#include <muda/cuda/cooperative_groups/memcpy_async.h>
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
void SubField::resize_data_buffer(size_t size, F&& func)
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
    resize_data_buffer(size,
                       [](std::byte* old_ptr, size_t old_size, std::byte* new_ptr, size_t new_size)
                       {
                           Memory()
                               .set(new_ptr + old_size, new_size - old_size, 0)  // set the new memory to 0
                               .transfer(new_ptr, old_ptr, old_size);  // copy the old memory to the new memory
                       });
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

MUDA_INLINE uint32_t SubField::round_up(uint32_t x, uint32_t n)
{
    MUDA_ASSERT((n & (n - 1)) == 0, "n is not power of 2");
    return (x + n - 1) & ~(n - 1);
}

MUDA_INLINE uint32_t SubField::align(uint32_t offset, uint32_t size, uint32_t min_alignment, uint32_t max_alignment)
{
    auto alignment = std::clamp(size, min_alignment, max_alignment);
    return round_up(offset, alignment);
}

MUDA_INLINE void SubField::build(const FieldBuildOptions& options)
{
    m_build_options = options;
    MUDA_ASSERT(!m_is_built, "Field is already built!");
    switch(m_layout.layout())
    {
        case FieldEntryLayout::AoSoA:
            build_aosoa(options);
            break;
        case FieldEntryLayout::SoA:
            build_soa(options);
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
    auto min_alignment = options.min_alignment;
    auto max_alignment = options.max_alignment;
    // we use the max alignment as the base array size
    auto base_array_size = max_alignment;
    // e.g. base array size = 4
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
    uint32_t struct_stride = 0;  // the stride of the "Struct"=> SoA total size
    for(auto e : m_entries)  // in an entry, the elem type is the same (e.g. float/int/double...)
    {
        // elem type = float/double/int ... or User Type
        auto elem_byte_size = e->elem_byte_size();
        // total elem count in innermost array:
        // scalar=1 vector3 = 3, vector4 = 4, matrix3x3 = 9, matrix4x4 = 16, and so on
        auto elem_count = e->shape().x * e->shape().y;
        struct_stride = align(struct_stride, elem_byte_size, min_alignment, max_alignment);
        auto total_elem_count_in_base_array = e->shape().x * e->shape().y * base_array_size;
        // now struct_stride is the offset of the entry in the "Struct"
        e->m_info.offset_in_struct = struct_stride;
        struct_stride += elem_byte_size * total_elem_count_in_base_array;
    }

    MUDA_ASSERT(struct_stride % base_array_size == 0,
                "m_struct_stride should be multiple of base_array_size");

    m_base_struct_stride = struct_stride;

    m_h_copy_map_buffer.reserve(4 * m_entries.size());
    for(size_t i = 0; i < m_entries.size(); ++i)
    {
        auto e                  = m_entries[i];
        e->m_info.struct_stride = m_struct_stride;
        e->m_name_ptr           = m_field.m_string_cache[e->m_name];
        auto btye_in_base_array = e->elem_byte_size() * options.max_alignment;  // the size of the entry in the base array
        auto first_comp_offset_in_base_struct = e->m_info.offset_in_base_struct;  // the offset of the entry in the base struct
        auto comp_count = e->shape().x * e->shape().y;
        for(int i = 0; i < comp_count; ++i)
        {
            details::SoACopyMap copy_map;
            copy_map.offset_in_base_struct =
                first_comp_offset_in_base_struct + i * btye_in_base_array;
            copy_map.btye_in_base_array = btye_in_base_array;
            m_h_copy_map_buffer.push_back(copy_map);
        }
    }
    // copy to device
    m_copy_map_buffer = m_h_copy_map_buffer;
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
        e->m_info.offset_in_base_struct = struct_stride;

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

MUDA_INLINE void SubField::resize(size_t num_elements)
{
    switch(m_layout.layout())
    {
        case FieldEntryLayout::AoSoA:
            resize_aosoa(num_elements);
            break;
        case FieldEntryLayout::SoA:
            resize_soa(num_elements);
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
    size_t outer_size = round_up(num_elements, m_layout.innermost_array_size());
    copy_resize_data_buffer(outer_size * m_struct_stride);
}


namespace details
{
    void soa_map_copy(BufferView<SoACopyMap> copy_maps,
                      size_t                 base_strcut_stride,
                      uint32_t               old_count_of_base,
                      uint32_t               new_count_of_base,
                      std::byte*             old_ptr,
                      std::byte*             new_ptr)
    {
        namespace cg = cooperative_groups;
        Memory().set(new_ptr, new_count_of_base * base_strcut_stride, 0);
        ParallelFor(LIGHT_WORKLOAD_BLOCK_SIZE)
            .apply(old_count_of_base,
                   [old_ptr,
                    new_ptr,
                    old_count_of_base,
                    new_count_of_base,
                    copy_maps = copy_maps.viewer()] __device__(int i) mutable
                   {
                       auto copy = [] __device__(std::byte * dst, const std::byte* src, size_t size)
                       {
                           using CopyType = int;
                           // floor to 4 bytes (like int/float ...)
                           size_t floor = size & ~sizeof(CopyType);
                           size_t rest  = size - floor;
                           auto   count = floor / sizeof(CopyType);
                           for(int i = 0; i < count; ++i)
                               ((CopyType*)dst)[i] = ((const CopyType*)src)[i];
                           // copy the rest
                           for(int i = 0; i < rest; ++i)
                               dst[floor + i] = src[floor + i];
                       };

                       for(int j = 0; j < copy_maps.dim(); ++j)
                       {
                           auto map = copy_maps(j);
                           auto old_offset =
                               map.offset_in_base_struct * old_count_of_base + i;
                           auto new_offset =
                               map.offset_in_base_struct * new_count_of_base + i;
                           copy(new_ptr + new_offset, old_ptr + old_offset, map.btye_in_base_array);
                       }
                   });
    }
}  // namespace details

MUDA_INLINE void SubField::resize_soa(size_t num_elements)
{
    auto base              = m_build_options.max_alignment;
    auto old_count_of_base = (m_num_elements + base - 1) / base;
    auto new_count_of_base = (num_elements + base - 1) / base;
    auto rounded_num       = new_count_of_base * base;
    m_struct_stride        = m_base_struct_stride * new_count_of_base;


    for(auto e : m_entries)
    {
        e->m_info.struct_stride = m_struct_stride;
        e->m_info.offset_in_struct = e->m_info.offset_in_base_struct * new_count_of_base;
        e->m_info.elem_count_based_stride = e->m_info.elem_byte_size * base * new_count_of_base;
    }

    resize_data_buffer(rounded_num,
                       [&](std::byte* old_ptr, size_t old_size, std::byte* new_ptr, size_t new_size)
                       {
                           details::soa_map_copy(m_copy_map_buffer,
                                                 m_base_struct_stride,
                                                 old_count_of_base,
                                                 new_count_of_base,
                                                 old_ptr,
                                                 new_ptr);
                       });
}

MUDA_INLINE void SubField::resize_aos(size_t num_elements)
{
    copy_resize_data_buffer(num_elements);
}

}  // namespace muda