namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewerBase : public FieldEntryViewerCore<IsConst, T, Layout, M, N>
{
    using Base = FieldEntryViewerCore<IsConst, T, Layout, M, N>;

    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    using ConstViewer    = FieldEntryViewerBase<true, T, Layout, M, N>;
    using NonConstViewer = FieldEntryViewerBase<false, T, Layout, M, N>;
    using ThisViewer     = FieldEntryViewerBase<IsConst, T, Layout, M, N>;

    using ConstMatrixMap = typename Base::ConstMatMap;
    using ThisMatrixMap  = typename Base::ThisMatMap;

    using Base::Base;

    MUDA_GENERIC FieldEntryViewerBase(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return ConstViewer{this->m_core}; }
    MUDA_GENERIC operator ConstViewer() const { return as_const(); }

    MUDA_GENERIC auto_const_t<T>* data(int i, int row_index, int col_index)
    {
        return Base::data(i, row_index, col_index);
    }

    MUDA_GENERIC const T* data(int i, int row_index, int col_index) const
    {
        return remove_const(this)->data(i, row_index, col_index);
    }

    MUDA_GENERIC auto operator()(int i)
    {
        return ThisMatrixMap{data(i, 0, 0), this->m_stride};
    }

    MUDA_GENERIC auto operator()(int i) const
    {
        return ConstMatrixMap{data(i, 0, 0), this->m_stride};
    }
};

template <typename T, FieldEntryLayout Layout, int M, int N>
class CFieldEntryViewer : public FieldEntryViewerBase<true, T, Layout, M, N>
{
    using Base = FieldEntryViewerBase<true, T, Layout, M, N>;

    MUDA_VIEWER_COMMON_NAME(CFieldEntryViewer);

  public:
    using Base::Base;
    using ConstViewer    = CFieldEntryViewer<T, Layout, M, N>;
    using NonConstViewer = FieldEntryViewer<T, Layout, M, N>;
    using ThisViewer     = CFieldEntryViewer<T, Layout, M, N>;

    MUDA_GENERIC CFieldEntryViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return *this; }
    MUDA_GENERIC operator ConstViewer() const { return as_const(); }

    using Base::operator();
};


template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntryViewer : public FieldEntryViewerBase<false, T, Layout, M, N>
{
    using Base = FieldEntryViewerBase<false, T, Layout, M, N>;

    MUDA_VIEWER_COMMON_NAME(FieldEntryViewer);

  public:
    using Base::Base;
    using ConstViewer    = CFieldEntryViewer<T, Layout, M, N>;
    using NonConstViewer = FieldEntryViewer<T, Layout, M, N>;
    using ThisViewer     = FieldEntryViewer<T, Layout, M, N>;

    MUDA_GENERIC FieldEntryViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC ConstViewer as_const() const
    {
        return ConstViewer{Base::as_const()};
    }
    MUDA_GENERIC operator ConstViewer() const { return as_const(); }

    using Base::operator();
};
}  // namespace muda