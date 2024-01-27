#pragma once
#include <muda/ext/field/sub_field_interface.h>

namespace muda
{
namespace details
{
    struct SoACopyMap
    {
        uint32_t offset_in_base_struct;
        uint32_t elem_byte_size;
    };
}  // namespace details


template <>
class SubFieldImpl<FieldEntryLayout::SoA> : public SubFieldInterface
{
    friend class SubField;

    DeviceBuffer<details::SoACopyMap> m_copy_map_buffer;
    std::vector<details::SoACopyMap>  m_h_copy_map_buffer;
    uint32_t                          m_base_struct_stride = ~0;

  protected:
    virtual void build() override;
    virtual void resize(size_t num_elements) override;

  public:
    using SubFieldInterface::SubFieldInterface;
    virtual ~SubFieldImpl() override = default;
};

}  // namespace muda

#include "details/soa_sub_field.inl"