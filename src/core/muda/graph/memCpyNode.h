#pragma once
#include "graph_base.h"

namespace muda
{
class memcpyNode : public graphNode
{
  public:
    using this_type = memcpyNode;
    friend class graph;
};
}  // namespace muda