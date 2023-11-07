#pragma once
#include <memory>
#include <muda/tools/host_device_string_cache.h>
#include <muda/field/field_build_options.h>
#include <muda/buffer/device_buffer.h>
#include <muda/field/field_entry_type.h>
#include <muda/field/field_builder.h>
#include <muda/field/sub_field_interface.h>

namespace muda
{
class Field;
class FieldEntryBase;
template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntry;

class SubField
{
    template <typename T>
    using U = std::unique_ptr<T>;

    Field&               m_field;
    std::string          m_name;
    U<SubFieldInterface> m_interface;
    bool                 m_is_built = false;
    std::byte* data_buffer() const { return m_interface->m_data_buffer; }


    FieldEntryBase* find_entry(std::string_view name) const;

    template <typename FieldEntryT>
    FieldEntryT* find_entry(std::string_view name) const;

  public:
    SubField(Field& field, std::string_view name);
    ~SubField();

    std::string_view name() const { return m_name; }

    size_t size() const;
    size_t num_entries() const { return m_interface->m_entries.size(); }
    void   resize(size_t num_elements);

    template <FieldEntryLayout Layout>
    FieldBuilder<Layout> builder(FieldEntryLayoutInfo layout = FieldEntryLayoutInfo{Layout});
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
    FieldBuilder<FieldEntryLayout::AoSoA> AoSoA(uint32_t innermost_array_size = 32);
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

    // delete copy and move
    SubField(const SubField&)            = delete;
    SubField(SubField&&)                 = delete;
    SubField& operator=(const SubField&) = delete;
    SubField& operator=(SubField&&)      = delete;

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
};
}  // namespace muda

#include "details/sub_field.inl"