#pragma once
#include <memory>
#include <muda/tools/host_device_string_cache.h>
#include <muda/field/field_build_options.h>
#include <muda/buffer/device_buffer.h>
#include <muda/field/field_entry_type.h>
#include <muda/field/field_builder.h>

namespace muda
{
class Field;
class FieldEntryBase;
template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntry;

class SubField
{
    Field&                                     m_field;
    std::string                                m_name;
    size_t                                     m_size = 0;
    std::vector<FieldEntryBase*>               m_entries;
    std::unordered_map<std::string, size_t>    m_name_to_index;
    FieldEntryLayoutInfo                       m_layout;
    FieldBuildOptions                          m_build_options;
    mutable DeviceBuffer<FieldEntryViewerBase> m_entries_buffer;

    mutable std::byte* m_data_buffer      = nullptr;
    size_t             m_data_buffer_size = 0;

    bool     m_is_built           = false;
    uint32_t m_struct_stride      = ~0;
    uint32_t m_base_struct_stride = ~0;
    size_t   m_num_elements       = 0;

    SubField(Field& field, std::string_view name);
    ~SubField();

    FieldEntryBase* SubField::find_entry(std::string_view name) const;

  public:
    template <FieldEntryLayout Layout>
    FieldBuilder<Layout> builder(FieldEntryLayoutInfo layout = FieldEntryLayoutInfo{});
    /// <summary>
    /// The layout is determined at runtime.
    /// </summary>
    /// <param name="layout"></param>
    /// <returns></returns>
    FieldBuilder<FieldEntryLayout::RuntimeLayout> builder(FieldEntryLayoutInfo layout);
    /// <summary>
    /// The layout is array of structs of arrays (determined at compile time)
    /// </summary>
    /// <param name="layout"></param>
    /// <returns></returns>
    FieldBuilder<FieldEntryLayout::AoSoA> AoSoA(FieldEntryLayoutInfo layout = FieldEntryLayoutInfo{});
    /// <summary>
    /// The layout is struct of arrays (determined at compile time)
    /// </summary>
    /// <returns></returns>
    FieldBuilder<FieldEntryLayout::SoA> SoA();
    /// <summary>
    /// The layout is array of structs (determined at compile time)
    /// </summary>
    /// <returns></returns>
    FieldBuilder<FieldEntryLayout::AoS> AoS();


    /// <summary>
    /// Finish building the field.
    /// </summary>
    /// <param name="options"></param>
    void build(const FieldBuildOptions& options = {});

    std::string_view name() const { return m_name; }
    size_t           size() const { return m_num_elements; }
    size_t           num_entries() const { return m_entries.size(); }
    size_t           struct_stride() const;
    void             resize(size_t num_elements);

  private:
    friend class Field;
    friend class SubField;
    template <FieldEntryLayout Layout>
    friend class FieldBuilder;
    friend class FieldEntryBase;
    template <typename T, FieldEntryLayout Layout, int M, int N>
    friend class FieldEntry;

    template <typename T, FieldEntryLayout Layout, int M, int N>
    FieldEntry<T, Layout, M, N>& create_entry(std::string_view     name,
                                              FieldEntryLayoutInfo layout,
                                              FieldEntryType       type,
                                              uint2                shape);
    void                         copy_resize_data_buffer(size_t size);
    template <typename F>  // F: void(std::byte* old_ptr, size_t old_size, std::byte* new_ptr, size_t new_size)
    void resize_data_buffer(size_t size, F&& func);

    void upload_entries() const;
    void resize_aosoa(size_t num_elements);
    void resize_soa(size_t num_elements);
    void resize_aos(size_t num_elements);

    void build_aosoa(const FieldBuildOptions& options);
    void build_soa(const FieldBuildOptions& options);
    void build_aos(const FieldBuildOptions& options);
    void resize_build_soa(size_t num_elements);

    static uint32_t round_up(uint32_t total, uint32_t N);
    static uint32_t align(uint32_t offset, uint32_t size, uint32_t min_alignment, uint32_t max_alignment);
};
}  // namespace muda

#include "details/sub_field.inl"