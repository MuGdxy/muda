#pragma once
#include <cuda.h>
#include <muda/muda_def.h>
#include <type_traits>

namespace muda
{
template <typename F>
class Kernel
{
    dim3         m_grid_dim;
    dim3         m_block_dim;
    size_t       m_shared_memory_size;
    cudaStream_t m_stream;
    F            m_kernel;

  public:
    MUDA_GENERIC Kernel(dim3 grid_dim, dim3 m_block_dim, size_t shared_memory_size, cudaStream_t stream, F f)
        : m_grid_dim(grid_dim)
        , m_block_dim(m_block_dim)
        , m_shared_memory_size(shared_memory_size)
        , m_stream(stream)
        , m_kernel(f)
    {
    }

    MUDA_GENERIC Kernel(F f)
        : Kernel{1, 1, 0, 0, f}
    {
    }

    MUDA_GENERIC Kernel(dim3 grid_dim, dim3 m_block_dim, F f)
        : Kernel{grid_dim, m_block_dim, 0, 0, f}
    {
    }

    MUDA_GENERIC Kernel(dim3 grid_dim, dim3 m_block_dim, size_t shared_memory_size, F f)
        : Kernel{grid_dim, m_block_dim, shared_memory_size, 0, f}
    {
    }

    MUDA_GENERIC Kernel(dim3 grid_dim, dim3 m_block_dim, cudaStream_t stream, F f)
        : Kernel{grid_dim, m_block_dim, 0, stream, f}
    {
    }

    MUDA_GENERIC Kernel(cudaStream_t stream, F f)
        : Kernel{1, 1, 0, stream, f}
    {
    }

    template <typename... Args>
    MUDA_GENERIC void operator()(Args&&... args) &&
    {
        static_assert(std::is_invocable_v<F, Args...>, "invalid arguments");
        m_kernel<<<m_grid_dim, m_block_dim, m_shared_memory_size, m_stream>>>(
            std::forward<Args>(args)...);
        checkCudaErrors(cudaGetLastError());
    }

    // delete copy and move
    MUDA_GENERIC         Kernel(const Kernel&)    = delete;
    MUDA_GENERIC Kernel& operator=(const Kernel&) = delete;
    MUDA_GENERIC         Kernel(Kernel&&)         = delete;
    MUDA_GENERIC Kernel& operator=(Kernel&&)      = delete;
};
}  // namespace muda
