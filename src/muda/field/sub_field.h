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
    Field&                                  m_field;
    std::string                             m_name;
    size_t                                  m_size = 0;
    std::vector<FieldEntryBase*>            m_entries;
    std::unordered_map<std::string, size_t> m_name_to_index;
    FieldEntryLayoutInfo                    m_layout;

    mutable DeviceBuffer<FieldEntryViewerBase> m_entries_buffer;
    mutable DeviceBuffer<std::byte>            m_data_buffer;

    bool     m_is_built      = false;
    uint32_t m_struct_stride = ~0;
    size_t   m_num_elements  = 0;

    SubField(Field& field, std::string_view name);
    ~SubField();

    FieldEntryBase* SubField::find_entry(std::string_view name) const;

  public:
    template <FieldEntryLayout Layout>
    FieldBuilder<Layout> builder(FieldEntryLayoutInfo layout = FieldEntryLayoutInfo{});
    FieldBuilder<FieldEntryLayout::RuntimeLayout> builder(FieldEntryLayoutInfo layout);
    FieldBuilder<FieldEntryLayout::AoSoA> AoSoA(FieldEntryLayoutInfo layout = FieldEntryLayoutInfo{});
    FieldBuilder<FieldEntryLayout::SoA> SoA();
    FieldBuilder<FieldEntryLayout::AoS> AoS();


    // finish building up the field
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

    void upload_entries() const;
    void resize_aosoa(size_t num_elements);
    void build_aosoa(const FieldBuildOptions& options);

    static uint32_t div_round_up(uint32_t total, uint32_t N);
    static uint32_t align(uint32_t offset, uint32_t size, uint32_t min_alignment, uint32_t max_alignment);
};
}  // namespace muda

#include "details/sub_field.inl"