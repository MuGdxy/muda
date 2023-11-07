#pragma once
#include <string>
#include <cinttypes>
#include <muda/tools/string_pointer.h>
#include <muda/buffer/device_buffer.h>
#include <muda/field/field_entry_layout.h>
#include <muda/field/field_entry_type.h>
#include <muda/field/field_entry_base_data.h>
#include <muda/field/field_entry_viewer.h>
namespace muda
{
class SubField;
class SubFieldInterface;
template <FieldEntryLayout Layout>
class SubFieldImpl;

class FieldEntryBase
{
  public:
    FieldEntryBase(SubField&            field,
                   FieldEntryLayoutInfo layout_info,
                   FieldEntryType       type,
                   uint2                shape,
                   uint32_t             m_elem_byte_size,
                   std::string_view     name)
        : m_field{field}
        , m_name{name}
    {
        m_info.layout_info    = layout_info;
        m_info.type           = type;
        m_info.shape          = shape;
        m_info.elem_byte_size = m_elem_byte_size;
    }
    ~FieldEntryBase() = default;

    auto name() const { return std::string{m_name}; }
    auto elem_byte_size() const { return m_info.elem_byte_size; }
    auto count() const { return m_info.elem_count; }

    auto                 struct_stride() const { return m_info.struct_stride; }
    auto                 layout() const { return m_info.layout_info.layout(); }
    auto                 layout_info() const { return m_info.layout_info; }
    auto                 type() const { return m_info.type; }
    auto                 shape() const { return m_info.shape; }
    FieldEntryViewerBase viewer();

  protected:
    friend class SubField;
    friend class SubFieldInterface;
    template <FieldEntryLayout Layout>
    friend class SubFieldImpl;

    // delete copy
    FieldEntryBase(const FieldEntryBase&)            = delete;
    FieldEntryBase& operator=(const FieldEntryBase&) = delete;

    SubField&              m_field;
    FieldEntryBaseData     m_info;
    std::string            m_name;
    details::StringPointer m_name_ptr;
};

constexpr int FieldEntryDynamicSize = -1;

template <typename T, FieldEntryLayout Layout, int M = 1, int N = 1>
class FieldEntry : public FieldEntryBase
{
    static_assert(M > 0 && N > 0, "M and N must be positive");

  public:
    FieldEntry(SubField& field, FieldEntryLayoutInfo layout, FieldEntryType type, std::string_view name)
        : FieldEntryBase{field,
                         layout,
                         type,
                         make_uint2(static_cast<uint32_t>(M), static_cast<uint32_t>(N)),
                         sizeof(T),
                         name}
    {
    }
    FieldEntry(SubField& field, FieldEntryLayoutInfo layout, FieldEntryType type, uint2 shape, std::string_view name)
        : FieldEntryBase{field, layout, type, shape, sizeof(T), name}
    {
    }
    FieldEntryViewer<T, Layout, M, N>  viewer();
    CFieldEntryViewer<T, Layout, M, N> cviewer() const;
};

template <typename T, FieldEntryLayout Layout>
class FieldEntry<T, Layout, FieldEntryDynamicSize, 1> : public FieldEntryBase
{
  public:
    FieldEntry(SubField& field, FieldEntryLayoutInfo layout, FieldEntryType type, uint32_t N, std::string_view name)
        : FieldEntryBase{field, layout, type, make_uint2(N, 1), sizeof(T), name}
    {
    }
    FieldEntry(SubField& field, FieldEntryLayoutInfo layout, FieldEntryType type, uint2 shape, std::string_view name)
        : FieldEntryBase{field, layout, type, shape, sizeof(T), name}
    {
    }
};

template <typename T, FieldEntryLayout Layout>
class FieldEntry<T, Layout, FieldEntryDynamicSize, FieldEntryDynamicSize> : public FieldEntryBase
{
  public:
    FieldEntry(SubField& field, FieldEntryLayoutInfo layout, FieldEntryType type, uint2 shape, std::string_view name)
        : FieldEntryBase{field, layout, type, shape, sizeof(T), name}
    {
    }
};

}  // namespace muda

#include "details/field_entry.inl"