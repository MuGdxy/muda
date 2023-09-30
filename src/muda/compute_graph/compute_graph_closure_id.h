#pragma once
#include <muda/tools/id_with_type.h>
namespace muda
{
class ClosureId : public U64IdWithType
{
    using U64IdWithType::U64IdWithType;
};
}  // namespace muda