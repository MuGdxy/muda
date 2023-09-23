#pragma once
#include <muda/exception.h>
#include <muda/compute_graph/compute_graph.h>

namespace muda
{
template <typename T>
LaunchBase<T>::LaunchBase(cudaStream_t stream)
    : m_stream(stream)
{
    if(ComputeGraphBuilder::is_phase_serial_launching())
    {
        if(stream != nullptr)
            throw muda::logic_error("LaunchBase: stream must be nullptr(default) in serial launching phase");
        init_stream(details::ComputeGraphAccessor().current_stream());
    }
}

}  // namespace muda