namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewBase : public FieldEntryViewCore<IsConst, T, Layout, M, N>
{
    using Base = FieldEntryViewCore<IsConst, T, Layout, M, N>;

    template <typename U>
    using auto_const_t = Base::template auto_const_t<U>;

  public:
    using Base::Base;
    using ConstView    = FieldEntryViewBase<true, T, Layout, M, N>;
    using NonConstView = FieldEntryViewBase<false, T, Layout, M, N>;
    using ThisView     = FieldEntryViewBase<IsConst, T, Layout, M, N>;
    using ElementType  = Eigen::Matrix<T, M, N>;


    MUDA_GENERIC FieldEntryViewBase(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return ConstView{m_core}; }
    MUDA_GENERIC operator ConstView() const { return as_const(); }


    MUDA_GENERIC auto_const_t<T>* data(int i, int row_index, int col_index)
    {
        return Base::data(i, row_index, col_index);
    }

    MUDA_GENERIC const T* data(int i, int row_index, int col_index) const
    {
        return remove_const(this)->data(i, row_index, col_index);
    }
};

template <typename T, FieldEntryLayout Layout, int M, int N>
class CFieldEntryView : public FieldEntryViewBase<true, T, Layout, M, N>
{
    using Base = FieldEntryViewBase<true, T, Layout, M, N>;

  public:
    using Base::Base;
    using ConstView    = CFieldEntryView<T, Layout, M, N>;
    using NonConstView = FieldEntryView<T, Layout, M, N>;
    using ThisView     = CFieldEntryView<T, Layout, M, N>;
    using ElementType  = typename Base::ElementType;

    MUDA_GENERIC CFieldEntryView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return *this; }
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_HOST void copy_to(BufferView<ElementType> dst) const;
};


template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryView : public FieldEntryViewBase<false, T, Layout, M, N>
{
    using Base = FieldEntryViewBase<false, T, Layout, M, N>;

  public:
    using Base::Base;
    using ConstView    = CFieldEntryView<T, Layout, M, N>;
    using NonConstView = FieldEntryView<T, Layout, M, N>;
    using ThisView     = FieldEntryView<T, Layout, M, N>;
    using ElementType  = typename Base::ElementType;

    MUDA_GENERIC FieldEntryView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC ConstView as_const() const
    {
        return ConstView{Base::as_const()};
    }
    MUDA_GENERIC operator ConstView() const { return as_const(); }


    template <FieldEntryLayout SrcLayout>
    MUDA_HOST void copy_from(CFieldEntryView<T, SrcLayout, M, N> src);
    MUDA_HOST void copy_from(CBufferView<ElementType> src);
    MUDA_HOST void copy_to(BufferView<ElementType> src) const
    {
        as_const().copy_to(src);
    }
    MUDA_HOST void fill(const ElementType& value);
};
}  // namespace muda

#include <muda/ext/field/field_entry_launch.h>

namespace muda
{
template <typename T, FieldEntryLayout Layout, int M, int N>
template <FieldEntryLayout SrcLayout>
MUDA_HOST void FieldEntryView<T, Layout, M, N>::copy_from(CFieldEntryView<T, SrcLayout, M, N> src)
{
    FieldEntryLaunch()  //
        .copy(*this, src)
        .wait();
}

template <typename T, FieldEntryLayout Layout, int M, int N>
MUDA_HOST void FieldEntryView<T, Layout, M, N>::copy_from(CBufferView<ElementType> src)
{
    FieldEntryLaunch()  //
        .copy(*this, src)
        .wait();
}

template <typename T, FieldEntryLayout Layout, int M, int N>
MUDA_HOST void CFieldEntryView<T, Layout, M, N>::copy_to(BufferView<ElementType> dst) const
{
    FieldEntryLaunch()  //
        .copy(dst, *this)
        .wait();
}

template <typename T, FieldEntryLayout Layout, int M, int N>
MUDA_HOST void FieldEntryView<T, Layout, M, N>::fill(const ElementType& value)
{
    FieldEntryLaunch()  //
        .fill(*this, value)
        .wait();
}
}  // namespace muda