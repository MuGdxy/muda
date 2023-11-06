#pragma once
#include <muda/viewer/viewer_base.h>
#include <muda/field/field_entry_layout.h>
#include <muda/field/field_entry_base_data.h>

namespace muda
{
template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntry;

class FieldEntryViewerBase : public ViewerBase
{
  public:
    MUDA_GENERIC FieldEntryViewerBase() {}

    MUDA_GENERIC FieldEntryViewerBase(std::byte*                buffer,
                                      const FieldEntryBaseData& info,
                                      details::StringPointer    name_ptr)
        : m_buffer(buffer)
        , m_info(info)
        , m_name_ptr(name_ptr)
    {
    }

  protected:
    FieldEntryLayoutInfo m_layout = {};

    std::byte*             m_buffer = nullptr;
    FieldEntryBaseData     m_info;
    details::StringPointer m_name_ptr = {};

    template <typename T>
    MUDA_GENERIC T& cast(std::byte* data);
    template <typename T>
    MUDA_GENERIC const T& cast(const std::byte* data) const;

    MUDA_GENERIC uint32_t aosoa_inner_index(int i) const;
    MUDA_GENERIC std::byte* aosoa_struct_begin(int i) const;
    MUDA_GENERIC std::byte* aosoa_elem_addr(int i) const;
    MUDA_GENERIC std::byte* aosoa_elem_addr(int i, int j) const;
    MUDA_GENERIC std::byte* aosoa_elem_addr(int i, int row_index, int col_index) const;

  public:
    MUDA_GENERIC auto layout() const { return m_layout; }
    MUDA_GENERIC auto count() const { return m_info.elem_count; }
    MUDA_GENERIC auto elem_byte_size() const { return m_info.elem_byte_size; }
    MUDA_GENERIC auto elem_alignment() const { return m_info.elem_alignment; }
    MUDA_GENERIC auto shape() const { return m_info.shape; }
    MUDA_GENERIC auto struct_stride() const { return m_info.struct_stride; }
    MUDA_GENERIC auto name() const { return m_name_ptr.auto_select(); }
};

// forward declaration
template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewer;
// implementation is in details/entry_viewers/ ...

/// <summary>
/// For MapMatrix e.g. Eigen::Map< ... >
/// </summary>
template <typename T, int M, int N>
class MatrixMapInfo
{
  public:
    T*  begin;
    int outer_stride;
    int inner_stride;
};
}  // namespace muda

#include "details/field_entry_viewer.inl"
#include "details/entry_viewers/aosoa_viewer.inl"
#include "details/entry_viewers/runtime_layout_viewer.inl"