#pragma once
#include <muda/debug.h>
#include <muda/exception.h>
#include <muda/compute_graph/compute_graph.h>

namespace muda
{
template <typename T>
LaunchBase<T>::LaunchBase(cudaStream_t stream) MUDA_NOEXCEPT : m_stream(stream)
{
    if(ComputeGraphBuilder::is_phase_serial_launching())
    {
        MUDA_ASSERT(stream == nullptr
                        || stream == details::ComputeGraphAccessor().current_stream(),
                    "LaunchBase: stream must be nullptr or equals to current stream");
        init_stream(details::ComputeGraphAccessor().current_stream());
    }
}

template <typename T>
LaunchBase<T>::~LaunchBase() MUDA_NOEXCEPT
{
#if MUDA_CHECK_ON
    if(ComputeGraphBuilder::is_direct_launching() && Debug::is_debug_sync_all())
        wait();
#endif
}
}  // namespace muda