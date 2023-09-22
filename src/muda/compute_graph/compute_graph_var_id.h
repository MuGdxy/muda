#pragma once
#include <muda/tools/id_with_type.h>
namespace muda
{
class VarId : public IdWithType
{
    using IdWithType::IdWithType;
};
}  // namespace muda