#define MUDA_SCALAR_ENTRY_ACCESSOR(i)                                          \
    {                                                                          \
        return cast<T>(aosoa_elem_addr(i));                                    \
    }

#define MUDA_VECTOR_ENTRY_ACCESSOR(i, comp)                                    \
    {                                                                          \
        return cast<T>(aosoa_elem_addr(i, comp));                              \
    }

#define MUDA_MATRIX_ENTRY_ACCESSOR(i, row_index, col_index)                    \
    {                                                                          \
        return cast<T>(aosoa_elem_addr(i, row_index, col_index));              \
    }

#include "entry_viewer_common.inl"
MUDA_ENTRY_VIEWER_COMMON(FieldEntryLayout::AoSoA,
                         MUDA_SCALAR_ENTRY_ACCESSOR,
                         MUDA_VECTOR_ENTRY_ACCESSOR,
                         MUDA_MATRIX_ENTRY_ACCESSOR)

#undef MUDA_ENTRY_VIEWER_COMMON
#undef MUDA_SCALAR_ENTRY_ACCESSOR
#undef MUDA_VECTOR_ENTRY_ACCESSOR
#undef MUDA_MATRIX_ENTRY_ACCESSOR
