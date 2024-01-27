#pragma once
#include <muda/tools/string_pointer.h>
#include <muda/view/view_base.h>
#include <muda/ext/field/field_entry_layout.h>
#include <muda/ext/field/field_entry_base_data.h>
#include <muda/ext/field/matrix_map_info.h>
#include <muda/ext/field/field_entry_core.h>
#include <muda/buffer/buffer_view.h>
#include <muda/ext/field/field_entry_viewer.h>

namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewCore : public ViewBase<IsConst>
{
    using Base = ViewBase<IsConst>;

  public:
    template <typename T>
    using auto_const_t = typename Base::auto_const_t<T>;

    using ConstViewer    = CFieldEntryViewer<T, Layout, M, N>;
    using NonConstViewer = FieldEntryViewer<T, Layout, M, N>;
    using ThisViewer = std::conditional_t<IsConst, ConstViewer, NonConstViewer>;

  protected:
    FieldEntryCore m_core;

    MUDA_GENERIC T* data(int i) const
    {
        return m_core.template data<T, Layout>(i);
    }

    MUDA_GENERIC T* data(int i, int j) const
    {
        return m_core.template data<T, Layout>(i, j);
    }

    MUDA_GENERIC T* data(int i, int row_index, int col_index) const
    {
        return m_core.template data<T, Layout>(i, row_index, col_index);
    }

  public:
    MUDA_GENERIC FieldEntryViewCore() = default;
    MUDA_GENERIC FieldEntryViewCore(const FieldEntryCore& core)
        : m_core{core}
    {
    }

    MUDA_GENERIC auto layout_info() const { return m_core.layout_info(); }
    MUDA_GENERIC auto layout() const { return layout_info().layout(); }
    MUDA_GENERIC auto count() const { return m_core.count(); }
    MUDA_GENERIC auto elem_byte_size() const { return m_core.elem_byte_size(); }
    MUDA_GENERIC auto shape() const { return m_core.shape(); }
    MUDA_GENERIC auto struct_stride() const { return m_core.struct_stride(); }

    // only available on host
    MUDA_HOST std::string_view name() const { return m_core.name(); }
    MUDA_GENERIC auto          viewer() { return ThisViewer{m_core}; }
    MUDA_GENERIC auto          cviewer() const { return ConstViewer{m_core}; }
};
}  // namespace muda

namespace muda
{
// forward declaration
template <bool IsConst, typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewBase;
template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryView;
template <typename T, FieldEntryLayout Layout, int M, int N>
class CFieldEntryView;
}  // namespace muda

// implementation
#include "details/entry_view/field_entry_view_matrix.inl"
#include "details/entry_view/field_entry_view_vector.inl"
#include "details/entry_view/field_entry_view_scalar.inl"
