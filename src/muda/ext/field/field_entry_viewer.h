#pragma once
#include <muda/viewer/viewer_base.h>
#include <muda/ext/field/field_entry_layout.h>
#include <muda/ext/field/field_entry_base_data.h>
#include <muda/ext/field/field_entry_core.h>
#include <muda/ext/field/matrix_map_info.h>
#include <muda/tools/host_device_config.h>
#include <Eigen/Core>

namespace muda
{
namespace details::field
{
    using MatStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    template <typename T, FieldEntryLayout Layout, int M, int N>
    MUDA_GENERIC MatStride make_stride(const FieldEntryCore& core)
    {
        MatStride ret;
        if constexpr(M == 1 && N == 1)
        {
            ret = MatStride{0, 0};
        }
        else if constexpr(N == 1)  // vector
        {
            auto begin = core.data<T, Layout>(0, 0);
            auto next  = core.data<T, Layout>(0, 1);
            ret        = MatStride{0, next - begin};
        }
        else  // matrix
        {
            auto begin      = core.data<T, Layout>(0, 0, 0);
            auto inner_next = core.data<T, Layout>(0, 1, 0);
            auto outer_next = core.data<T, Layout>(0, 0, 1);
            ret             = MatStride{outer_next - begin, inner_next - begin};
        }
        return ret;
    }
}  // namespace details::field

template <bool IsConst, typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewerCore : protected ViewerBase<IsConst>
{
    using Base = ViewerBase<IsConst>;

  public:
    using MatStride = details::field::MatStride;

    using ConstMatMap = Eigen::Map<const Eigen::Matrix<T, M, N>, 0, MatStride>;
    using NonConstMatMap = Eigen::Map<Eigen::Matrix<T, M, N>, 0, MatStride>;
    using ThisMatMap = std::conditional_t<IsConst, ConstMatMap, NonConstMatMap>;

  protected:
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

    HostDeviceConfigView<FieldEntryCore> m_core;
    MatStride                            m_stride;
    int                                  m_offset = 0;
    int                                  m_size   = 0;

  public:
    MUDA_GENERIC FieldEntryViewerCore() {}

    MUDA_GENERIC FieldEntryViewerCore(HostDeviceConfigView<FieldEntryCore> core, int offset, int size)
        : m_core(core)
        , m_offset(offset)
        , m_size(size)
    {
        Base::name(core->name_string_pointer());

        MUDA_KERNEL_ASSERT(m_offset >= 0 && m_size >= 0 && m_offset + m_size <= total_count(),
                           "FieldEntryViewer[%s:%s]: offset/size indexing out of range, size=%d, offset=%d, size=%d",
                           this->name(),
                           this->kernel_name(),
                           this->total_count(),
                           m_offset,
                           m_size);

        m_stride = details::field::make_stride<T, Layout, M, N>(*m_core);
    }

    MUDA_GENERIC FieldEntryViewerCore(const FieldEntryViewerCore&) = default;

    // here we don't care about the const/non-const T* access
    // we will impl that in the derived class
    MUDA_GENERIC T* data(int i) const
    {
        check_index(i);
        return m_core->template data<T, Layout>(m_offset + i);
    }

    MUDA_GENERIC T* data(int i, int j) const
    {
        check_index(i);

        MUDA_KERNEL_ASSERT(j < shape().x,
                           "FieldEntry[%s:%s]: vector component indexing out of range, shape=(%d, %d), index=%d",
                           this->name(),
                           this->kernel_name(),
                           shape().x,
                           shape().y,
                           j);
        return m_core->template data<T, Layout>(m_offset + i, j);
    }

    MUDA_GENERIC T* data(int i, int row_index, int col_index) const
    {
        check_index(i);

        MUDA_KERNEL_ASSERT(row_index < shape().x && col_index < shape().y,
                           "FieldEntry[%s:%s]: vector component indexing out of range, shape=(%d,%d), index=(%d,%d)",
                           this->name(),
                           this->kernel_name(),
                           shape().x,
                           shape().y,
                           row_index,
                           col_index);
        return m_core->template data<T, Layout>(m_offset + i, row_index, col_index);
    }

  public:
    MUDA_GENERIC auto layout_info() const { return m_core->layout_info(); }
    MUDA_GENERIC auto layout() const { return m_core->layout(); }
    MUDA_GENERIC auto offset() const { return m_offset; }
    MUDA_GENERIC auto size() const { return m_size; }
    MUDA_GENERIC auto total_count() const { return m_core->count(); }
    MUDA_GENERIC auto elem_byte_size() const
    {
        return m_core->elem_byte_size();
    }
    MUDA_GENERIC auto shape() const { return m_core->shape(); }
    MUDA_GENERIC auto struct_stride() const { return m_core->struct_stride(); }
    MUDA_GENERIC auto entry_name() const { return m_core->name(); }

  private:
    MUDA_INLINE MUDA_GENERIC void check_index(int i) const
    {
        MUDA_KERNEL_ASSERT(i < m_size,
                           "FieldEntryViewer[%s:%s]: indexing out of range, index=%d, size=%d, offset=%d, entry_total_count=%d",
                           this->name(),
                           this->kernel_name(),
                           i,
                           m_size,
                           m_offset,
                           this->total_count());
    }
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
