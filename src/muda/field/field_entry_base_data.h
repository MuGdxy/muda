#pragma once
#include <muda/field/field_entry_type.h>
#include <muda/field/field_entry_layout.h>
namespace muda
{
class FieldEntryBaseData
{
  public:
    FieldEntryLayoutInfo layout;
    FieldEntryType       type           = FieldEntryType::None;
    uint32_t             offset_in_struct          = ~0;
    uint32_t             elem_byte_size = ~0;
    uint32_t             elem_alignment = ~0;
    uint32_t             elem_count     = ~0;
    uint2                shape;
    uint32_t             struct_stride = ~0;
};
}  // namespace muda