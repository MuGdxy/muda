namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout, int N>
class FieldEntryViewBase<IsConst, T, Layout, N, 1>
    : public FieldEntryViewCore<IsConst, T, Layout, N, 1>
{
    using Base = FieldEntryViewCore<IsConst, T, Layout, N, 1>;

    template <typename U>
    using auto_const_t = Base::template auto_const_t<U>;

  public:
    using Base::Base;
    using ConstView    = FieldEntryViewBase<true, T, Layout, N, 1>;
    using NonConstView = FieldEntryViewBase<false, T, Layout, N, 1>;
    using ThisView     = FieldEntryViewBase<IsConst, T, Layout, N, 1>;
    using ElementType  = Eigen::Vector<T, N>;


    MUDA_GENERIC auto as_const() const { return ConstView{m_core}; }

    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_GENERIC auto_const_t<T>* data(int i, int comp_j)
    {
        return Base::data(i, comp_j);
    }

    MUDA_GENERIC const T* data(int i, int comp_j) const
    {
        return remove_const(this)->data(i, comp_j);
    }
};

template <typename T, FieldEntryLayout Layout, int N>
class CFieldEntryView<T, Layout, N, 1>
    : public FieldEntryViewBase<true, T, Layout, N, 1>
{
    using Base = FieldEntryViewBase<true, T, Layout, N, 1>;

  public:
    using Base::Base;
    using ConstView    = CFieldEntryView<T, Layout, N, 1>;
    using NonConstView = FieldEntryView<T, Layout, N, 1>;
    using ThisView     = CFieldEntryView<T, Layout, N, 1>;
    using ElementType  = typename Base::ElementType;

    MUDA_GENERIC CFieldEntryView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return *this; }
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_HOST void copy_to(BufferView<ElementType> dst) const;
};

template <typename T, FieldEntryLayout Layout, int N>
class FieldEntryView<T, Layout, N, 1>
    : public FieldEntryViewBase<false, T, Layout, N, 1>
{
    using Base = FieldEntryViewBase<false, T, Layout, N, 1>;

  public:
    using Base::Base;
    using ConstView    = CFieldEntryView<T, Layout, N, 1>;
    using NonConstView = FieldEntryView<T, Layout, N, 1>;
    using ThisView     = FieldEntryView<T, Layout, N, 1>;
    using ElementType  = typename Base::ElementType;

    MUDA_GENERIC FieldEntryView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return ConstView{Base::as_const()}; }
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    template <FieldEntryLayout SrcLayout>
    MUDA_HOST void copy_from(CFieldEntryView<T, SrcLayout, N, 1> src);
    MUDA_HOST void copy_from(CBufferView<ElementType> src);
    MUDA_HOST void copy_to(BufferView<ElementType> dst) const
    {
        as_const().copy_to(dst);
    }

    MUDA_HOST void fill(const ElementType& value);
};
}  // namespace muda

#include <muda/ext/field/field_entry_launch.h>

namespace muda
{
template <typename T, FieldEntryLayout Layout, int N>
template <FieldEntryLayout SrcLayout>
MUDA_HOST void FieldEntryView<T, Layout, N, 1>::copy_from(CFieldEntryView<T, SrcLayout, N, 1> src)
{
    FieldEntryLaunch()  //
        .copy(*this, src)
        .wait();
}

template <typename T, FieldEntryLayout Layout, int N>
MUDA_HOST void CFieldEntryView<T, Layout, N, 1>::copy_to(BufferView<ElementType> dst) const
{
    FieldEntryLaunch()  //
        .copy(dst, *this)
        .wait();
}

template <typename T, FieldEntryLayout Layout, int N>
MUDA_HOST void FieldEntryView<T, Layout, N, 1>::copy_from(CBufferView<ElementType> src)
{
    FieldEntryLaunch()  //
        .copy(*this, src)
        .wait();
}

template <typename T, FieldEntryLayout Layout, int N>
MUDA_HOST void FieldEntryView<T, Layout, N, 1>::fill(const ElementType& value)
{
    FieldEntryLaunch()  //
        .fill(*this, value)
        .wait();
}
}  // namespace muda