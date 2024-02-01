#pragma once
#include <string>
#include <cinttypes>
#include <muda/tools/string_pointer.h>
#include <muda/buffer/device_buffer.h>
#include <muda/ext/field/field_entry_type.h>
#include <muda/ext/field/field_entry_base_data.h>
#include <muda/ext/field/field_entry_view.h>

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

    auto struct_stride() const { return m_info.struct_stride; }
    auto layout() const { return m_info.layout_info.layout(); }
    auto layout_info() const { return m_info.layout_info; }
    auto type() const { return m_info.type; }
    auto shape() const { return m_info.shape; }
    // FieldEntryViewerBase viewer();

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

template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntry : public FieldEntryBase
{
    static_assert(M > 0 && N > 0, "M and N must be positive");

  public:
    using ElementType = typename FieldEntryView<T, Layout, M, N>::ElementType;

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

    FieldEntryView<T, Layout, M, N> view()
    {
        MUDA_ASSERT(m_field.data_buffer() != nullptr, "Resize the field before you use it!");
        return FieldEntryView<T, Layout, M, N>{
            FieldEntryCore{m_field.data_buffer(), m_info, m_name_ptr},
            0,
            static_cast<int>(m_info.elem_count)};
    }

    CFieldEntryView<T, Layout, M, N> view() const
    {
        MUDA_ASSERT(m_field.data_buffer() != nullptr, "Resize the field before you use it!");
        return CFieldEntryView<T, Layout, M, N>{
            FieldEntryCore{m_field.data_buffer(), m_info, m_name_ptr},
            0,
            static_cast<int>(m_info.elem_count)};
    }

    auto view(int offset) { return view().subview(offset); }
    auto view(int offset) const { return view().subview(offset); }

    auto view(int offset, int count) { return view().subview(offset, count); }
    auto view(int offset, int count) const
    {
        return view().subview(offset, count);
    }

    FieldEntryViewer<T, Layout, M, N>  viewer() { return view().viewer(); }
    CFieldEntryViewer<T, Layout, M, N> cviewer() const
    {
        return view().viewer();
    }

    void copy_to(DeviceBuffer<ElementType>& dst) const;
    void copy_to(std::vector<ElementType>& dst) const;

    void copy_from(const DeviceBuffer<ElementType>& src);
    void copy_from(const std::vector<ElementType>& src);

    template <FieldEntryLayout SrcLayout>
    void copy_from(const FieldEntry<T, SrcLayout, M, N>& src);

    void fill(const ElementType& value);

  private:
    mutable DeviceBuffer<ElementType> m_workpace;  // for data copy, if needed
};

constexpr int FieldEntryDynamicSize = -1;

template <typename T, FieldEntryLayout Layout>
class FieldEntry<T, Layout, FieldEntryDynamicSize, 1> : public FieldEntryBase
{
  public:
    FieldEntry(SubField& field, FieldEntryLayoutInfo layout, FieldEntryType type, uint32_t N, std::string_view name)
        : FieldEntryBase{field, layout, type, make_uint2(N, 1), sizeof(T), name}
    {
        MUDA_ERROR_WITH_LOCATION("Not implemented yet");
    }
};

template <typename T, FieldEntryLayout Layout>
class FieldEntry<T, Layout, FieldEntryDynamicSize, FieldEntryDynamicSize> : public FieldEntryBase
{
  public:
    FieldEntry(SubField& field, FieldEntryLayoutInfo layout, FieldEntryType type, uint2 shape, std::string_view name)
        : FieldEntryBase{field, layout, type, shape, sizeof(T), name}
    {
        MUDA_ERROR_WITH_LOCATION("Not implemented yet");
    }
};
}  // namespace muda

#include "details/field_entry.inl"