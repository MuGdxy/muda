// don't place #pragma once at the beginning of this file
// because it should be inserted in multiple files

#define MUDA_ENTRY_VIEWER_COMMON(MUDA_FIELD_ENTRY_LAYOUT,                                   \
                                 MUDA_SCALAR_ENTRY_ACCESSOR,                                \
                                 MUDA_VECTOR_ENTRY_ACCESSOR,                                \
                                 MUDA_MATRIX_ENTRY_ACCESSOR)                                \
    namespace muda                                                                          \
    {                                                                                       \
    template <typename T>                                                                   \
    class CFieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, 1, 1> : public FieldEntryViewerBase \
    {                                                                                       \
        friend class FieldEntry<T, MUDA_FIELD_ENTRY_LAYOUT, 1, 1>;                          \
        friend class FieldViewer;                                                           \
                                                                                            \
      public:                                                                               \
        using FieldEntryViewerBase::FieldEntryViewerBase;                                   \
        CFieldEntryViewer(const std::byte*          buffer,                                 \
                          const FieldEntryBaseData& info,                                   \
                          details::StringPointer    name_ptr)                               \
            : FieldEntryViewerBase(const_cast<std::byte*>(buffer), info, name_ptr)          \
        {                                                                                   \
        }                                                                                   \
                                                                                            \
        CFieldEntryViewer(const FieldEntryViewerBase& base)                                 \
            : FieldEntryViewerBase{base}                                                    \
        {                                                                                   \
        }                                                                                   \
                                                                                            \
        MUDA_GENERIC const T& operator()(int i) const MUDA_SCALAR_ENTRY_ACCESSOR(i);        \
    };                                                                                      \
                                                                                            \
    template <typename T>                                                                   \
    class FieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, 1, 1> : public FieldEntryViewerBase  \
    {                                                                                       \
        friend class FieldEntry<T, MUDA_FIELD_ENTRY_LAYOUT, 1, 1>;                          \
        friend class FieldViewer;                                                           \
                                                                                            \
      public:                                                                               \
        using FieldEntryViewerBase::FieldEntryViewerBase;                                   \
                                                                                            \
        FieldEntryViewer(const FieldEntryViewerBase& base)                                  \
            : FieldEntryViewerBase{base}                                                    \
        {                                                                                   \
        }                                                                                   \
                                                                                            \
        operator CFieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, 1, 1>() const                \
        {                                                                                   \
            return CFieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, 1, 1>{                     \
                static_cast<const FieldEntryViewerBase&>(*this)};                           \
        }                                                                                   \
                                                                                            \
        MUDA_GENERIC T& operator()(int i) MUDA_SCALAR_ENTRY_ACCESSOR(i);                    \
        MUDA_GENERIC const T& operator()(int i) const MUDA_SCALAR_ENTRY_ACCESSOR(i);        \
    };                                                                                      \
                                                                                            \
    template <typename T, int N>                                                            \
    class CFieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, N, 1> : public FieldEntryViewerBase \
    {                                                                                       \
        using MapInfo = CMatrixMapInfo<T, N, 1>;                                            \
                                                                                            \
        friend class FieldEntry<T, MUDA_FIELD_ENTRY_LAYOUT, N, 1>;                          \
        friend class FieldViewer;                                                           \
                                                                                            \
      public:                                                                               \
        using FieldEntryViewerBase::FieldEntryViewerBase;                                   \
                                                                                            \
        CFieldEntryViewer(const FieldEntryViewerBase& base)                                 \
            : FieldEntryViewerBase{base}                                                    \
        {                                                                                   \
        }                                                                                   \
                                                                                            \
                                                                                            \
        MUDA_GENERIC const T& x(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(i, 0);              \
        MUDA_GENERIC const T& y(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(i, 1);              \
        MUDA_GENERIC const T& z(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(i, 2);              \
        MUDA_GENERIC const T& w(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(i, 3);              \
                                                                                            \
        MUDA_GENERIC const T& operator()(int i, int comp) const                             \
            MUDA_VECTOR_ENTRY_ACCESSOR(i, comp);                                            \
                                                                                            \
        MUDA_GENERIC auto operator()(int i) const                                           \
        {                                                                                   \
            auto begin = &(*this)(i, 0);                                                    \
            auto next  = &(*this)(i, 1);                                                    \
            return MapInfo{begin, 0, int(next - begin)};                                    \
        }                                                                                   \
    };                                                                                      \
                                                                                            \
    template <typename T, int N>                                                            \
    class FieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, N, 1> : public FieldEntryViewerBase  \
    {                                                                                       \
        using MapInfo = MatrixMapInfo<T, N, 1>;                                             \
                                                                                            \
        friend class FieldEntry<T, MUDA_FIELD_ENTRY_LAYOUT, N, 1>;                          \
        friend class FieldViewer;                                                           \
                                                                                            \
      public:                                                                               \
        using FieldEntryViewerBase::FieldEntryViewerBase;                                   \
                                                                                            \
        FieldEntryViewer(const FieldEntryViewerBase& base)                                  \
            : FieldEntryViewerBase{base}                                                    \
        {                                                                                   \
        }                                                                                   \
                                                                                            \
        operator CFieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, N, 1>() const                \
        {                                                                                   \
            return CFieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, N, 1>{                     \
                static_cast<const FieldEntryViewerBase&>(*this)};                           \
        }                                                                                   \
                                                                                            \
        MUDA_GENERIC T& x(int i) MUDA_VECTOR_ENTRY_ACCESSOR(i, 0);                          \
        MUDA_GENERIC T& y(int i) MUDA_VECTOR_ENTRY_ACCESSOR(i, 1);                          \
        MUDA_GENERIC T& z(int i) MUDA_VECTOR_ENTRY_ACCESSOR(i, 2);                          \
        MUDA_GENERIC T& w(int i) MUDA_VECTOR_ENTRY_ACCESSOR(i, 3);                          \
                                                                                            \
        MUDA_GENERIC const T& x(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(i, 0);              \
        MUDA_GENERIC const T& y(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(i, 1);              \
        MUDA_GENERIC const T& z(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(i, 2);              \
        MUDA_GENERIC const T& w(int i) const MUDA_VECTOR_ENTRY_ACCESSOR(i, 3);              \
                                                                                            \
        MUDA_GENERIC T& operator()(int i, int comp) MUDA_VECTOR_ENTRY_ACCESSOR(i, comp);    \
        MUDA_GENERIC const T& operator()(int i, int comp) const                             \
            MUDA_VECTOR_ENTRY_ACCESSOR(i, comp);                                            \
                                                                                            \
        MUDA_GENERIC auto operator()(int i) const                                           \
        {                                                                                   \
            auto begin = &(*this)(i, 0);                                                    \
            auto next  = &(*this)(i, 1);                                                    \
            return MapInfo{const_cast<T*>(begin), 0, int(next - begin)};                    \
        }                                                                                   \
    };                                                                                      \
                                                                                            \
    template <typename T, int M, int N>                                                     \
    class CFieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, M, N> : public FieldEntryViewerBase \
    {                                                                                       \
        using MapInfo = CMatrixMapInfo<T, M, N>;                                            \
                                                                                            \
        friend class FieldEntry<T, MUDA_FIELD_ENTRY_LAYOUT, M, N>;                          \
        friend class FieldViewer;                                                           \
                                                                                            \
      public:                                                                               \
        using FieldEntryViewerBase::FieldEntryViewerBase;                                   \
                                                                                            \
        CFieldEntryViewer(const FieldEntryViewerBase& base)                                 \
            : FieldEntryViewerBase{base}                                                    \
        {                                                                                   \
        }                                                                                   \
                                                                                            \
        MUDA_GENERIC const T& operator()(int i, int row_index, int col_index) const         \
            MUDA_MATRIX_ENTRY_ACCESSOR(i, row_index, col_index);                            \
                                                                                            \
        MUDA_GENERIC auto operator()(int i) const                                           \
        {                                                                                   \
            auto begin      = &(*this)(i, 0, 0);                                            \
            auto inner_next = &(*this)(i, 1, 0);                                            \
            auto outer_next = &(*this)(i, 0, 1);                                            \
            return MapInfo{begin, int(outer_next - begin), int(inner_next - begin)};        \
        }                                                                                   \
    };                                                                                      \
                                                                                            \
    template <typename T, int M, int N>                                                     \
    class FieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, M, N> : public FieldEntryViewerBase  \
    {                                                                                       \
        using MapInfo = MatrixMapInfo<T, M, N>;                                             \
                                                                                            \
        friend class FieldEntry<T, MUDA_FIELD_ENTRY_LAYOUT, M, N>;                          \
        friend class FieldViewer;                                                           \
                                                                                            \
      public:                                                                               \
        using FieldEntryViewerBase::FieldEntryViewerBase;                                   \
                                                                                            \
        FieldEntryViewer(const FieldEntryViewerBase& base)                                  \
            : FieldEntryViewerBase{base}                                                    \
        {                                                                                   \
        }                                                                                   \
                                                                                            \
        operator CFieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, M, N>() const                \
        {                                                                                   \
            return CFieldEntryViewer<T, MUDA_FIELD_ENTRY_LAYOUT, M, N>{                     \
                static_cast<const FieldEntryViewerBase&>(*this)};                           \
        }                                                                                   \
                                                                                            \
                                                                                            \
        MUDA_GENERIC T& operator()(int i, int row_index, int col_index)                     \
            MUDA_MATRIX_ENTRY_ACCESSOR(i, row_index, col_index);                            \
        MUDA_GENERIC const T& operator()(int i, int row_index, int col_index) const         \
            MUDA_MATRIX_ENTRY_ACCESSOR(i, row_index, col_index);                            \
                                                                                            \
        MUDA_GENERIC auto operator()(int i) const                                           \
        {                                                                                   \
            auto begin      = &(*this)(i, 0, 0);                                            \
            auto inner_next = &(*this)(i, 1, 0);                                            \
            auto outer_next = &(*this)(i, 0, 1);                                            \
            return MatrixMapInfo<T, M, N>{const_cast<T*>(begin),                            \
                                          int(outer_next - begin),                          \
                                          int(inner_next - begin)};                         \
        }                                                                                   \
    };                                                                                      \
    }
