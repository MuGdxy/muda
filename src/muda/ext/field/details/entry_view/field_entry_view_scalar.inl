namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout>
class FieldEntryViewBase<IsConst, T, Layout, 1, 1>
    : public FieldEntryViewCore<IsConst, T, Layout, 1, 1>
{
    using Base = FieldEntryViewCore<IsConst, T, Layout, 1, 1>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    using Base::Base;
    using ConstView    = FieldEntryViewBase<true, T, Layout, 1, 1>;
    using NonConstView = FieldEntryViewBase<false, T, Layout, 1, 1>;
    using ThisView     = FieldEntryViewBase<IsConst, T, Layout, 1, 1>;

    using ConstViewer    = CFieldEntryViewer<T, Layout, 1, 1>;
    using NonConstViewer = FieldEntryViewer<T, Layout, 1, 1>;
    using ThisViewer = std::conditional_t<IsConst, ConstViewer, NonConstViewer>;
    using ElementType = T;

    MUDA_GENERIC auto as_const() const
    {
        return ConstView{this->m_core, Base::offset(), Base::size()};
    }
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_GENERIC auto_const_t<T>* data(int i) { return Base::data(i); }
    MUDA_GENERIC const T*         data(int i) const
    {
        return remove_const(this)->data(i);
    }

    MUDA_GENERIC auto subview(int offset) const
    {
        return ConstView{this->m_core, this->m_offset + offset, this->m_size - offset};
    }

    MUDA_GENERIC auto subview(int offset, int size) const
    {
        return ConstView{this->m_core, this->m_offset + offset, size};
    }

    MUDA_GENERIC auto subview(int offset)
    {
        return ThisView{this->m_core, this->m_offset + offset, this->m_size - offset};
    }

    MUDA_GENERIC auto subview(int offset, int size)
    {
        return ThisView{this->m_core, this->m_offset + offset, size};
    }

    /**********************************************************************************
    * Entry View As Iterator
    ***********************************************************************************/

    // Random Access Iterator Interface
    using value_type        = T;
    using reference         = T&;
    using pointer           = T*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = size_t;

    MUDA_GENERIC ThisView operator+(int i)
    {
        return ThisView{this->m_core, this->m_offset + i, this->m_size - i, typename Base::AsIterator{}};
    }
    MUDA_GENERIC ConstView operator+(int i) const
    {
        return remove_const(*this).operator+(i).as_const();
    }
    MUDA_GENERIC reference operator*() { return *data(0); }
    MUDA_GENERIC auto_const_t<T>& operator[](int i) { return *data(i); }
    MUDA_GENERIC const T&         operator[](int i) const { return *data(i); }
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
    using value_type   = T;

    MUDA_GENERIC CFieldEntryView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return *this; }
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_GENERIC auto subview(int offset) const
    {
        return ConstView{Base::subview(offset)};
    }

    MUDA_GENERIC auto subview(int offset, int size) const
    {
        return ConstView{Base::subview(offset, size)};
    }

    MUDA_GENERIC auto subview(int offset)
    {
        return ThisView{Base::subview(offset)};
    }

    MUDA_GENERIC auto subview(int offset, int size)
    {
        return ThisView{Base::subview(offset, size)};
    }

    MUDA_HOST void copy_to(BufferView<T> dst) const;
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
    using value_type   = T;

    MUDA_GENERIC FieldEntryView(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return ConstView{Base::as_const()}; }
    MUDA_GENERIC operator ConstView() const { return as_const(); }

    MUDA_GENERIC auto subview(int offset) const
    {
        return ConstView{Base::subview(offset)};
    }

    MUDA_GENERIC auto subview(int offset, int size) const
    {
        return ConstView{Base::subview(offset, size)};
    }

    MUDA_GENERIC auto subview(int offset)
    {
        return ThisView{Base::subview(offset)};
    }

    MUDA_GENERIC auto subview(int offset, int size)
    {
        return ThisView{Base::subview(offset, size)};
    }

    template <FieldEntryLayout SrcLayout>
    MUDA_HOST void copy_from(CFieldEntryView<T, SrcLayout, 1, 1> src);
    MUDA_HOST void copy_from(CBufferView<T> src);

    MUDA_HOST void copy_to(BufferView<T> dst) const { as_const().copy_to(dst); }

    MUDA_HOST void fill(const T& value);
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
        .copy(*this, src)
        .wait();
}

template <typename T, FieldEntryLayout Layout>
MUDA_HOST void FieldEntryView<T, Layout, 1, 1>::copy_from(CBufferView<T> src)
{
    FieldEntryLaunch()  //
        .copy(*this, src)
        .wait();
}

template <typename T, FieldEntryLayout Layout>
MUDA_HOST void CFieldEntryView<T, Layout, 1, 1>::copy_to(BufferView<T> dst) const
{
    FieldEntryLaunch()  //
        .copy(dst, *this)
        .wait();
}

template <typename T, FieldEntryLayout Layout>
MUDA_HOST void FieldEntryView<T, Layout, 1, 1>::fill(const T& value)
{
    FieldEntryLaunch()  //
        .fill(*this, value)
        .wait();
}
}  // namespace muda