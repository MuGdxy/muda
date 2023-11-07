#pragma once
#include <muda/field/sub_field_interface.h>

namespace muda
{
template <>
class SubFieldImpl<FieldEntryLayout::AoSoA> : public SubFieldInterface
{
    friend class SubField;
  protected:

    virtual void build() override;
    virtual void resize(size_t num_elements) override;

  public:
    using SubFieldInterface::SubFieldInterface;
    virtual ~SubFieldImpl() override = default;
};

}  // namespace muda

#include "details/aosoa_sub_field.inl"