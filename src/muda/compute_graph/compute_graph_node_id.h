#pragma once
#include <muda/tools/id_with_type.h>
namespace muda
{
class NodeId : public IdWithType
{
    using IdWithType::IdWithType;
};
}  // namespace muda