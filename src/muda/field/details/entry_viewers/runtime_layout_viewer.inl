namespace muda
{
template <typename T>
class FieldEntryViewer<T, FieldEntryLayout::RuntimeLayout, 1, 1> : public FieldEntryViewerBase
{
    friend class FieldEntry<T, FieldEntryLayout::RuntimeLayout, 1, 1>;
    friend class FieldViewer;

  public:
    using FieldEntryViewerBase::FieldEntryViewerBase;

#define MUDA_SCALAR_ENTRY_ACCESSOR()                                           \
    {                                                                          \
        switch(m_layout.layout())                                              \
        {                                                                      \
            case FieldEntryLayout::AoSoA:                                      \
                return cast<T>(aosoa_elem_addr(i));                            \
            case FieldEntryLayout::AoS:                                        \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
            case FieldEntryLayout::SoA:                                        \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
            default:                                                           \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
        }                                                                      \
    }

    MUDA_GENERIC T&       operator()(int i) MUDA_SCALAR_ENTRY_ACCESSOR();
    MUDA_GENERIC const T& operator()(int i) const MUDA_SCALAR_ENTRY_ACCESSOR();
#undef MUDA_SCALAR_ENTRY_ACCESSOR
};

template <typename T, int N>
class FieldEntryViewer<T, FieldEntryLayout::RuntimeLayout, N, 1> : public FieldEntryViewerBase
{
    friend class FieldEntry<T, FieldEntryLayout::RuntimeLayout, N, 1>;
    friend class FieldViewer;

  public:
    using FieldEntryViewerBase::FieldEntryViewerBase;

#define MUDA_VECTOR_ENTRY_ACCESSOR(comp)                                       \
    {                                                                          \
        switch(m_layout.layout())                                              \
        {                                                                      \
            case FieldEntryLayout::AoSoA:                                      \
                return cast<T>(aosoa_elem_addr(i, comp));                      \
            case FieldEntryLayout::AoS:                                        \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
            case FieldEntryLayout::SoA:                                        \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
            default:                                                           \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
        }                                                                      \
    }

    MUDA_GENERIC T& x(int i) MUDA_VECTOR_ENTRY_ACCESSOR(0);
    MUDA_GENERIC T& y(int i) MUDA_VECTOR_ENTRY_ACCESSOR(1);
    MUDA_GENERIC T& z(int i) MUDA_VECTOR_ENTRY_ACCESSOR(2);
    MUDA_GENERIC T& w(int i) MUDA_VECTOR_ENTRY_ACCESSOR(3);

    MUDA_GENERIC const T& x(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(0);
    MUDA_GENERIC const T& y(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(1);
    MUDA_GENERIC const T& z(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(2);
    MUDA_GENERIC const T& w(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(3);

    MUDA_GENERIC T& operator()(int i, int comp) MUDA_VECTOR_ENTRY_ACCESSOR(comp);
    MUDA_GENERIC const T& operator()(int i, int comp) const
        MUDA_VECTOR_ENTRY_ACCESSOR(comp);

    MUDA_GENERIC MatrixMapInfo<T, N, 1> operator()(int i) const
    {
        auto begin = &(*this)(i, 0);
        auto next  = &(*this)(i, 1);
        return MatrixMapInfo<T, N, 1>{const_cast<T*>(begin), 0, int(next - begin)};
    }
#undef MUDA_VECTOR_ENTRY_ACCESSOR
};

template <typename T, int M, int N>
class FieldEntryViewer<T, FieldEntryLayout::RuntimeLayout, M, N> : public FieldEntryViewerBase
{
    using Type = FieldEntryViewer<T, FieldEntryLayout::RuntimeLayout, M, N>;
    MUDA_VIEWER_COMMON_NAME(Type);

    friend class FieldEntry<T, FieldEntryLayout::RuntimeLayout, M, N>;
    friend class FieldViewer;

  public:
    using FieldEntryViewerBase::FieldEntryViewerBase;

#define MUDA_MATRIX_ENTRY_ACCESSOR()                                           \
    {                                                                          \
        return cast<T>(aosoa_elem_addr(i, row_index, col_index));              \
    }

    MUDA_GENERIC T& operator()(int i, int row_index, int col_index)
        MUDA_MATRIX_ENTRY_ACCESSOR();
    MUDA_GENERIC const T& operator()(int i, int row_index, int col_index) const
        MUDA_MATRIX_ENTRY_ACCESSOR();

    MUDA_GENERIC MatrixMapInfo<T, M, N> operator()(int i) const
    {
        auto begin      = &(*this)(i, 0, 0);
        auto inner_next = &(*this)(i, 1, 0);
        auto outer_next = &(*this)(i, 0, 1);
        return MatrixMapInfo<T, M, N>{const_cast<T*>(begin),
                                      int(outer_next - begin),
                                      int(inner_next - begin)};
    }
#undef MUDA_SCALAR_ENTRY_ACCESSOR
};

}  // namespace muda