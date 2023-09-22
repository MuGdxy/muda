#pragma once

namespace muda
{
enum class ComputeGraphNodeType
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