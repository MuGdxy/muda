#include "entry_viewer_common.inl"

#define MUDA_SCALAR_ENTRY_ACCESSOR(i)                                          \
    {                                                                          \
        switch(layout())                                                       \
        {                                                                      \
            case FieldEntryLayout::AoSoA:                                      \
                return cast<T>(aosoa_elem_addr(i));                            \
            case FieldEntryLayout::AoS:                                        \
                return cast<T>(aos_elem_addr(i));                              \
            case FieldEntryLayout::SoA:                                        \
                return cast<T>(soa_elem_addr(i));                              \
            default:                                                           \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
        }                                                                      \
    }

#define MUDA_VECTOR_ENTRY_ACCESSOR(i, comp)                                    \
    {                                                                          \
        switch(layout())                                                       \
        {                                                                      \
            case FieldEntryLayout::AoSoA:                                      \
                return cast<T>(aosoa_elem_addr(i, comp));                      \
            case FieldEntryLayout::AoS:                                        \
                return cast<T>(aos_elem_addr(i, comp));                        \
            case FieldEntryLayout::SoA:                                        \
                return cast<T>(soa_elem_addr(i, comp));                        \
            default:                                                           \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
        }                                                                      \
    }

#define MUDA_MATRIX_ENTRY_ACCESSOR(i, row_index, col_index)                    \
    {                                                                          \
        switch(layout())                                                       \
        {                                                                      \
            case FieldEntryLayout::AoSoA:                                      \
                return cast<T>(aosoa_elem_addr(i, row_index, col_index));      \
            case FieldEntryLayout::AoS:                                        \
                return cast<T>(aos_elem_addr(i, row_index, col_index));        \
            case FieldEntryLayout::SoA:                                        \
                return cast<T>(soa_elem_addr(i, row_index, col_index));        \
            default:                                                           \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
        }                                                                      \
    }


MUDA_ENTRY_VIEWER_COMMON(FieldEntryLayout::RuntimeLayout,
                         MUDA_SCALAR_ENTRY_ACCESSOR,
                         MUDA_VECTOR_ENTRY_ACCESSOR,
                         MUDA_MATRIX_ENTRY_ACCESSOR)


#undef MUDA_SCALAR_ENTRY_ACCESSOR
#undef MUDA_VECTOR_ENTRY_ACCESSOR
#undef MUDA_MATRIX_ENTRY_ACCESSOR
#undef MUDA_ENTRY_VIEWER_COMMON
