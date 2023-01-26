#pragma once
#include "base.h"

namespace muda
{
class eventRecordNode : public graphNode
{
  public:
    using this_type = eventRecordNode;
    friend class graph;
};

class eventWaitNode : public graphNode
{
  public:
    using this_type = eventWaitNode;
    friend class graph;
};
}  // namespace muda