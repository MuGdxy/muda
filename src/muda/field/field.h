#pragma once
#include <memory>
#include <muda/tools/host_device_string_cache.h>
#include <muda/field/field_viewer.h>

namespace muda
{
class FieldEntryBase;
template <typename T, FieldEntryLayout Layout, int M, int N>
class FieldEntry;
class SubField;

class Field
{
    friend class SubField;
    details::HostDeviceStringCache          m_string_cache;
    std::vector<SubField*>                  m_sub_fields;
    std::unordered_map<std::string, size_t> m_name_to_index;

  public:
    Field();
    // sub field count
    size_t num_sub_fields() const { return m_sub_fields.size(); }

    FieldViewer viewer() const;
    // create or find a subfield
    SubField& operator[](std::string_view name);
    ~Field();
};
}  // namespace muda

#include "details/field.inl"