namespace muda
{
template <bool IsConst, typename T, FieldEntryLayout Layout>
class FieldEntryViewerBase<IsConst, T, Layout, 1, 1>
    : public FieldEntryViewerCore<IsConst, T, Layout, 1, 1>
{
    using Base = FieldEntryViewerCore<IsConst, T, Layout, 1, 1>;

    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    using Base::Base;
    using ConstViewer    = FieldEntryViewerBase<true, T, Layout, 1, 1>;
    using NonConstViewer = FieldEntryViewerBase<false, T, Layout, 1, 1>;
    using ThisViewer     = FieldEntryViewerBase<IsConst, T, Layout, 1, 1>;

    MUDA_GENERIC auto as_const() const { return ConstViewer{this->m_core}; }
    MUDA_GENERIC operator ConstViewer() const { return as_const(); }

    MUDA_GENERIC auto_const_t<T>* data(int i) { return Base::data(i); }
    MUDA_GENERIC const T*         data(int i) const
    {
        return remove_const(this)->data(i);
    }

    MUDA_GENERIC auto_const_t<T>& operator()(int i) { return *data(i); }
    MUDA_GENERIC const T&         operator()(int i) const { return *data(i); }
};

template <typename T, FieldEntryLayout Layout>
class CFieldEntryViewer<T, Layout, 1, 1>
    : public FieldEntryViewerBase<true, T, Layout, 1, 1>
{
    using Base = FieldEntryViewerBase<true, T, Layout, 1, 1>;
    
    MUDA_VIEWER_COMMON_NAME(CFieldEntryViewer);

  public:
    using Base::Base;
    using ConstViewer    = CFieldEntryViewer<T, Layout, 1, 1>;
    using NonConstViewer = FieldEntryViewer<T, Layout, 1, 1>;
    using ThisViewer     = CFieldEntryViewer<T, Layout, 1, 1>;

    MUDA_GENERIC CFieldEntryViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return *this; }
    MUDA_GENERIC operator ConstViewer() const { return as_const(); }

    using Base::operator();
};

template <typename T, FieldEntryLayout Layout>
class FieldEntryViewer<T, Layout, 1, 1>
    : public FieldEntryViewerBase<false, T, Layout, 1, 1>
{
    using Base = FieldEntryViewerBase<false, T, Layout, 1, 1>;

    MUDA_VIEWER_COMMON_NAME(FieldEntryViewer);

  public:
    using Base::Base;
    using ConstViewer    = CFieldEntryViewer<T, Layout, 1, 1>;
    using NonConstViewer = FieldEntryViewer<T, Layout, 1, 1>;
    using ThisViewer     = FieldEntryViewer<T, Layout, 1, 1>;

    MUDA_GENERIC FieldEntryViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC auto as_const() const { return ConstViewer{Base::as_const()}; }

    MUDA_GENERIC operator ConstViewer() const { return as_const(); }

    using Base::operator();
};
}  // namespace muda