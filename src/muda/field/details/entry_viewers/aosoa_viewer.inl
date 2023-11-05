namespace muda
{
template <typename T>
class FieldEntryViewer<T, FieldEntryLayout::AoSoA, 1, 1> : public FieldEntryViewerBase
{
    using Type = FieldEntryViewer<T, FieldEntryLayout::AoSoA, 1, 1>;
    MUDA_VIEWER_COMMON_NAME(Type);

    friend class FieldEntry<T, FieldEntryLayout::AoSoA, 1, 1>;
    friend class FieldViewer;
  public:
    using FieldEntryViewerBase::FieldEntryViewerBase;
#define MUDA_SCALAR_ENTRY_ACCESSOR()                                           \
    {                                                                          \
        return cast<T>(aosoa_elem_addr(i, 0));                                 \
    }

    MUDA_GENERIC T&       operator()(int i) MUDA_SCALAR_ENTRY_ACCESSOR();
    MUDA_GENERIC const T& operator()(int i) const MUDA_SCALAR_ENTRY_ACCESSOR();
#undef MUDA_SCALAR_ENTRY_ACCESSOR
};

template <typename T, int N>
class FieldEntryViewer<T, FieldEntryLayout::AoSoA, N, 1> : public FieldEntryViewerBase
{
    using Type = FieldEntryViewer<T, FieldEntryLayout::AoSoA, N, 1>;
    MUDA_VIEWER_COMMON_NAME(Type);

    friend class FieldEntry<T, FieldEntryLayout::AoSoA, N, 1>;
    friend class FieldViewer;
  public:
      using FieldEntryViewerBase::FieldEntryViewerBase;

#define MUDA_SCALAR_ENTRY_ACCESSOR(comp_j)                                     \
    {                                                                          \
        return cast<T>(aosoa_elem_addr(i, comp_j));                            \
    }
    MUDA_GENERIC T& x(int i) MUDA_SCALAR_ENTRY_ACCESSOR(0);
    MUDA_GENERIC T& y(int i) MUDA_SCALAR_ENTRY_ACCESSOR(1);
    MUDA_GENERIC T& z(int i) MUDA_SCALAR_ENTRY_ACCESSOR(2);
    MUDA_GENERIC T& w(int i) MUDA_SCALAR_ENTRY_ACCESSOR(3);

    MUDA_GENERIC const T& x(int i) const MUDA_SCALAR_ENTRY_ACCESSOR(0);
    MUDA_GENERIC const T& y(int i) const MUDA_SCALAR_ENTRY_ACCESSOR(1);
    MUDA_GENERIC const T& z(int i) const MUDA_SCALAR_ENTRY_ACCESSOR(2);
    MUDA_GENERIC const T& w(int i) const MUDA_SCALAR_ENTRY_ACCESSOR(3);

    MUDA_GENERIC T& operator()(int i, int comp_j) MUDA_SCALAR_ENTRY_ACCESSOR(comp_j);
    MUDA_GENERIC const T& operator()(int i, int comp_j) const
        MUDA_SCALAR_ENTRY_ACCESSOR(comp_j);

    MUDA_GENERIC VectorMapInfo<T, N> operator()(int i) const
    {
        auto begin = &(*this)(i, 0);
        auto next  = &(*this)(i, 1);
        return VectorMapInfo<T, N>{const_cast<T*>(begin), int(next - begin)};
    }
#undef MUDA_SCALAR_ENTRY_ACCESSOR
};
}  // namespace muda