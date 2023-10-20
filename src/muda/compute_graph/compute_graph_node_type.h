#pragma once
#include <cinttypes>
namespace muda
{
enum class ComputeGraphNodeType : uint8_t
{
    None,
    KernelNode,
    MemcpyNode,
    CaptureNode,
    EventRecordNode,
    EventWaitNode,
    Max
};
}  // namespace muda