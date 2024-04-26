#pragma once
#include <muda/tools/string_pointer.h>
#include <muda/view/view_base.h>
#include <muda/ext/field/field_entry_layout.h>
#include <muda/ext/field/field_entry_base_data.h>
#include <muda/ext/field/matrix_map_info.h>
#include <muda/ext/field/field_entry_core.h>
#include <muda/buffer/buffer_view.h>
#include <muda/ext/field/field_entry_viewer.h>
#include <muda/tools/host_device_config.h>

namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewCore : public ViewBase<IsConst>
{
    using Base       = ViewBase<IsConst>;
    using ViewerCore = FieldEntryViewerCore<IsConst, T, Layout, M, N>;
    friend class FieldEntryLaunch;

  public:
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

    using ConstViewer    = CFieldEntryViewer<T, Layout, M, N>;
    using NonConstViewer = FieldEntryViewer<T, Layout, M, N>;
    using ThisViewer = std::conditional_t<IsConst, ConstViewer, NonConstViewer>;

    using MatStride      = typename ViewerCore::MatStride;
    using ConstMatMap    = typename ViewerCore::ConstMatMap;
    using NonConstMatMap = typename ViewerCore::NonConstMatMap;
    using ThisMatMap     = typename ViewerCore::ThisMatMap;

  protected:
    HostDeviceConfigView<FieldEntryCore> m_core;
    MatStride                            m_stride;
    int                                  m_offset = 0;
    int                                  m_size   = 0;


    MUDA_GENERIC T* data(int i) const
    {
        return m_core->template data<T, Layout>(m_offset + i);
    }

    MUDA_GENERIC T* data(int i, int j) const
    {

        return m_core->template data<T, Layout>(m_offset + i, j);
    }

    MUDA_GENERIC T* data(int i, int row_index, int col_index) const
    {
        return m_core->template data<T, Layout>(m_offset + i, row_index, col_index);
    }

  protected:
    struct AsIterator
    {
    };

    MUDA_GENERIC FieldEntryViewCore(HostDeviceConfigView<FieldEntryCore> core, int offset, int size, AsIterator)
        : m_core{core}
        , m_offset{offset}
        , m_size{size}
    {
        m_stride = details::field::make_stride<T, Layout, M, N>(*m_core);
    }

  public:
    MUDA_GENERIC FieldEntryViewCore() = default;
    MUDA_GENERIC FieldEntryViewCore(HostDeviceConfigView<FieldEntryCore> core, int offset, int size)
        : FieldEntryViewCore(core, offset, size, AsIterator{})
    {
        MUDA_KERNEL_ASSERT(offset >= 0 && size >= 0 && offset + size <= core->count(),
                           "(offset,size) is out of range, offset=%d, size=%d, count=%d",
                           offset,
                           size,
                           core->count());
    }

    MUDA_GENERIC auto layout_info() const { return m_core->layout_info(); }
    MUDA_GENERIC auto layout() const { return layout_info().layout(); }
    MUDA_GENERIC auto offset() const { return m_offset; }
    MUDA_GENERIC auto size() const { return m_size; }
    MUDA_GENERIC auto total_count() const { return m_core->count(); }
    MUDA_GENERIC auto elem_byte_size() const
    {
        return m_core->elem_byte_size();
    }
    MUDA_GENERIC auto shape() const { return m_core->shape(); }
    MUDA_GENERIC auto struct_stride() const { return m_core->struct_stride(); }

    MUDA_GENERIC auto name() const { return m_core->name(); }
    MUDA_GENERIC auto viewer() { return ThisViewer{m_core, offset(), size()}; }
    MUDA_GENERIC auto cviewer() const
    {
        return ConstViewer{m_core, offset(), size()};
    }
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

namespace muda
{
template <typename T, FieldEntryLayout Layout, int M, int N>
struct read_only_viewer<FieldEntryView<T, Layout, M, N>>
{
    using type = CFieldEntryView<T, Layout, M, N>;
};

template <typename T, FieldEntryLayout Layout, int M, int N>
struct read_write_viewer<CFieldEntryView<T, Layout, M, N>>
{
    using type = FieldEntryView<T, Layout, M, N>;
};
}  // namespace muda
