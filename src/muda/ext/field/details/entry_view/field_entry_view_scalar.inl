namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout>
class FieldEntryViewBase<IsConst, T, Layout, 1, 1>
    : public FieldEntryViewCore<IsConst, T, Layout, 1, 1>
{
    using Base = FieldEntryViewCore<IsConst, T, Layout, 1, 1>;

    template <typename U>
    using auto_const_t = Base::template auto_const_t<U>;

  public:
    using Base::Base;
    using ConstView    = FieldEntryViewBase<true, T, Layout, 1, 1>;
    using NonConstView = FieldEntryViewBase<false, T, Layout, 1, 1>;
    using ThisView     = FieldEntryViewBase<IsConst, T, Layout, 1, 1>;

    using ConstViewer    = CFieldEntryViewer<T, Layout, 1, 1>;
    using NonConstViewer = FieldEntryViewer<T, Layout, 1, 1>;
    using ThisViewer = std::conditional_t<IsConst, ConstViewer, NonConstViewer>;
    using ElementType = T;

    MUDA_GENERIC auto as_const() const { return ConstView{m_core}; }
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_GENERIC auto_const_t<T>* data(int i) { return Base::data(i); }
    MUDA_GENERIC const T*         data(int i) const
    {
        return remove_const(this)->data(i);
    }
};

template <typename T, FieldEntryLayout Layout>
class CFieldEntryView<T, Layout, 1, 1>
    : public FieldEntryViewBase<true, T, Layout, 1, 1>
{
    using Base = FieldEntryViewBase<true, T, Layout, 1, 1>;

  public:
    using Base::Base;
    using ConstView    = CFieldEntryView<T, Layout, 1, 1>;
    using NonConstView = FieldEntryView<T, Layout, 1, 1>;
    using ThisView     = CFieldEntryView<T, Layout, 1, 1>;
    using ElementType  = typename Base::ElementType;

    MUDA_GENERIC CFieldEntryView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return *this; }
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_GENERIC void copy_to(BufferView<T> dst) const;
};


template <typename T, FieldEntryLayout Layout>
class FieldEntryView<T, Layout, 1, 1>
    : public FieldEntryViewBase<false, T, Layout, 1, 1>
{
    using Base = FieldEntryViewBase<false, T, Layout, 1, 1>;

  public:
    using Base::Base;
    using ConstView    = CFieldEntryView<T, Layout, 1, 1>;
    using NonConstView = FieldEntryView<T, Layout, 1, 1>;
    using ThisView     = FieldEntryView<T, Layout, 1, 1>;
    using ElementType  = typename Base::ElementType;

    MUDA_GENERIC FieldEntryView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return ConstView{Base::as_const()}; }

    MUDA_GENERIC operator ConstView() const { return as_const(); }

    template <FieldEntryLayout SrcLayout>
    MUDA_HOST void copy_from(CFieldEntryView<T, SrcLayout, 1, 1> src);
    MUDA_HOST void copy_from(CBufferView<T> src);

    MUDA_GENERIC void copy_to(BufferView<T> dst) const
    {
        as_const().copy_to(dst);
    }
};
}  // namespace muda

#include <muda/ext/field/field_entry_launch.h>

namespace muda
{
template <typename T, FieldEntryLayout Layout>
template <FieldEntryLayout SrcLayout>
MUDA_HOST void FieldEntryView<T, Layout, 1, 1>::copy_from(CFieldEntryView<T, SrcLayout, 1, 1> src)
{
    FieldEntryLaunch()  //
        .template copy<T, Layout, SrcLayout>(*this, src)
        .wait();
}

template <typename T, FieldEntryLayout Layout>
MUDA_HOST void FieldEntryView<T, Layout, 1, 1>::copy_from(CBufferView<T> src)
{
    FieldEntryLaunch()  //
        .template copy<T, Layout>(*this, src)
        .wait();
}

template <typename T, FieldEntryLayout Layout>
MUDA_HOST void CFieldEntryView<T, Layout, 1, 1>::copy_to(BufferView<T> dst) const
{
    FieldEntryLaunch()  //
        .template copy<T, Layout>(dst, *this)
        .wait();
}

}  // namespace muda