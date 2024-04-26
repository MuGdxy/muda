namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewBase : public FieldEntryViewCore<IsConst, T, Layout, M, N>
{
    using Base       = FieldEntryViewCore<IsConst, T, Layout, M, N>;
    using ViewerBase = FieldEntryViewerBase<IsConst, T, Layout, M, N>;

    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    using Base::Base;
    using ConstView    = FieldEntryViewBase<true, T, Layout, M, N>;
    using NonConstView = FieldEntryViewBase<false, T, Layout, M, N>;
    using ThisView     = FieldEntryViewBase<IsConst, T, Layout, M, N>;
    using ElementType  = Eigen::Matrix<T, M, N>;

    using ConstMatrixMap = typename Base::ConstMatMap;
    using ThisMatrixMap  = typename Base::ThisMatMap;


    MUDA_GENERIC FieldEntryViewBase(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const
    {
        return ConstView{this->m_core, Base::offset(), Base::size()};
    }
    MUDA_GENERIC operator ConstView() const { return as_const(); }


    MUDA_GENERIC auto_const_t<T>* data(int i, int row_index, int col_index)
    {
        return Base::data(i, row_index, col_index);
    }

    MUDA_GENERIC const T* data(int i, int row_index, int col_index) const
    {
        return remove_const(this)->data(i, row_index, col_index);
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

    class DummyPointer
    {
        ThisMatrixMap map;

      public:
        MUDA_GENERIC DummyPointer(ThisMatrixMap map)
            : map(map)
        {
        }
        MUDA_GENERIC auto operator*() { return map; }
    };

    // Random Access Iterator Interface
    using value_type        = ElementType;
    using reference         = ThisMatrixMap;
    using pointer           = DummyPointer;
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
    MUDA_GENERIC reference operator*() { return (*this)[0]; }

    MUDA_GENERIC auto operator[](int i)
    {
        return ThisMatrixMap{data(i, 0, 0), this->m_stride};
    }
    MUDA_GENERIC auto operator[](int i) const
    {
        return ConstMatrixMap{data(i, 0, 0), this->m_stride};
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