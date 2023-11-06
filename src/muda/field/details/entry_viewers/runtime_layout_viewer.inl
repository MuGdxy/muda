#define MUDA_SCALAR_ENTRY_ACCESSOR(i)                                          \
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

#define MUDA_VECTOR_ENTRY_ACCESSOR(i, comp)                                    \
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

#define MUDA_MATRIX_ENTRY_ACCESSOR(i, row_index, col_index)                    \
    {                                                                          \
        switch(m_layout.layout())                                              \
        {                                                                      \
            case FieldEntryLayout::AoSoA:                                      \
                return cast<T>(aosoa_elem_addr(i, row_index, col_index));      \
            case FieldEntryLayout::AoS:                                        \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
            case FieldEntryLayout::SoA:                                        \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
            default:                                                           \
                MUDA_KERNEL_ERROR_WITH_LOCATION("No impl yet");                \
        }                                                                      \
    }

#include "entry_viewer_common.inl"
MUDA_ENTRY_VIEWER_COMMON(FieldEntryLayout::RuntimeLayout,
                         MUDA_SCALAR_ENTRY_ACCESSOR,
                         MUDA_VECTOR_ENTRY_ACCESSOR,
                         MUDA_MATRIX_ENTRY_ACCESSOR)
#undef MUDA_ENTRY_VIEWER_COMMON

#undef MUDA_SCALAR_ENTRY_ACCESSOR
#undef MUDA_VECTOR_ENTRY_ACCESSOR
#undef MUDA_MATRIX_ENTRY_ACCESSOR
