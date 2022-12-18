#pragma once
#include "graph_base.h"

namespace muda
{
class memFreeNode : public graphNode
{
  public:
    using this_type = memFreeNode;
    friend class graph;
};
}  // namespace muda