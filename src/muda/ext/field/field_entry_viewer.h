#pragma once
#include <muda/viewer/viewer_base.h>
#include <muda/ext/field/field_entry_layout.h>
#include <muda/ext/field/field_entry_base_data.h>
#include <muda/ext/field/field_entry_core.h>
#include <muda/ext/field/matrix_map_info.h>
#include <Eigen/Core>

namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewerCore : protected ViewerBase<IsConst>
{
    using Base = ViewerBase<IsConst>;

  protected:
    template <typename T>
    using auto_const_t = Base::template auto_const_t<T>;

    using MatStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

    using ConstMatMap = Eigen::Map<const Eigen::Matrix<T, M, N>, 0, MatStride>;
    using NonConstMatMap = Eigen::Map<Eigen::Matrix<T, M, N>, 0, MatStride>;
    using ThisMatMap = std::conditional_t<IsConst, ConstMatMap, NonConstMatMap>;

    FieldEntryCore m_core;
    MatStride      m_stride;

  public:
    MUDA_GENERIC FieldEntryViewerCore() {}

    MUDA_GENERIC FieldEntryViewerCore(const FieldEntryCore& core)
        : m_core(core)
    {
        Base::name(core.name_string_pointer());

        if constexpr(M == 1 && N == 1)
        {
            m_stride = MatStride{0, 0};
        }
        else if constexpr(N == 1)  // vector
        {
            auto begin = core.data<T, Layout>(0, 0);
            auto next  = core.data<T, Layout>(0, 1);
            m_stride   = MatStride{0, next - begin};
        }
        else  // matrix
        {
            auto begin      = core.data<T, Layout>(0, 0, 0);
            auto inner_next = core.data<T, Layout>(0, 1, 0);
            auto outer_next = core.data<T, Layout>(0, 0, 1);
            m_stride        = MatStride{outer_next - begin, inner_next - begin};
        }
    }

    MUDA_GENERIC FieldEntryViewerCore(const FieldEntryViewerCore&) = default;

    // here we don't care about the const/non-const T* access
    // we will impl that in the derived class
    MUDA_GENERIC T* data(int i) const
    {
        MUDA_KERNEL_ASSERT(i < count(),
                           "FieldEntry[%s:%s]: count indexing out of range, size=%d, index=%d",
                           kernel_name(),
                           name(),
                           count(),
                           i);
        return m_core.template data<T, Layout>(i);
    }

    MUDA_GENERIC T* data(int i, int j) const
    {
        MUDA_KERNEL_ASSERT(i < count(),
                           "FieldEntry[%s:%s]: count indexing out of range, size=%d, index=%d",
                           kernel_name(),
                           name(),
                           count(),
                           i);

        MUDA_KERNEL_ASSERT(j < shape().x,
                           "FieldEntry[%s:%s]: vector component indexing out of range, shape=(%d, %d), index=%d",
                           kernel_name(),
                           name(),
                           shape().x,
                           shape().y,
                           j);
        return m_core.template data<T, Layout>(i, j);
    }

    MUDA_GENERIC T* data(int i, int row_index, int col_index) const
    {
        MUDA_KERNEL_ASSERT(i < count(),
                           "FieldEntry[%s:%s]: count indexing out of range, size=%d, index=%d",
                           kernel_name(),
                           name(),
                           count(),
                           i);

        MUDA_KERNEL_ASSERT(row_index < shape().x && col_index < shape().y,
                           "FieldEntry[%s:%s]: vector component indexing out of range, shape=(%d,%d), index=(%d,%d)",
                           kernel_name(),
                           name(),
                           shape().x,
                           shape().y,
                           row_index,
                           col_index);
        return m_core.template data<T, Layout>(i, row_index, col_index);
    }

  public:
    MUDA_GENERIC auto layout_info() const { return m_core.layout_info(); }
    MUDA_GENERIC auto layout() const { return m_core.layout(); }
    MUDA_GENERIC auto count() const { return m_core.count(); }
    MUDA_GENERIC auto elem_byte_size() const { return m_core.elem_byte_size(); }
    MUDA_GENERIC auto shape() const { return m_core.shape(); }
    MUDA_GENERIC auto struct_stride() const { return m_core.struct_stride(); }
    MUDA_GENERIC auto entry_name() const { return m_core.name(); }
};
}  // namespace muda


namespace muda
{
// forward declaration
template <bool IsConst, typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewerBase;
template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewer;
template <typename T, FieldEntryLayout Layout, int M, int N>
class CFieldEntryViewer;
}  // namespace muda

// implementation
#include "details/entry_viewers/field_entry_viewer_matrix.inl"
#include "details/entry_viewers/field_entry_viewer_vector.inl"
#include "details/entry_viewers/field_entry_viewer_scalar.inl"
