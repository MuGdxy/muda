#pragma once
#include <cuda_runtime.h>
#include <muda/check/check.h>
namespace muda::details
{
class ByteTempBuffer
{
  public:
    ByteTempBuffer() {}
    ~ByteTempBuffer()
    {
        if(m_data)
        {
            // we don't check the error here to prevent exception when app is shutting down
            cudaFree(m_data);
        }
    }

    ByteTempBuffer(ByteTempBuffer&& other) noexcept
    {
        m_size           = other.m_size;
        m_capacity       = other.m_capacity;
        m_data           = other.m_data;
        other.m_size     = 0;
        other.m_capacity = 0;
        other.m_data     = nullptr;
    }

    ByteTempBuffer& operator=(ByteTempBuffer&& other) noexcept
    {
        if(this == &other)
        {
            return *this;
        }
        m_size           = other.m_size;
        m_capacity       = other.m_capacity;
        m_data           = other.m_data;
        other.m_size     = 0;
        other.m_capacity = 0;
        other.m_data     = nullptr;
        return *this;
    }

    // no change on copy
    ByteTempBuffer(const ByteTempBuffer&) noexcept {}
    // no change on copy
    ByteTempBuffer& operator=(const ByteTempBuffer&) noexcept { return *this; }

    void resize(cudaStream_t stream, size_t size)
    {
        if(size > m_capacity)
        {
            if(m_data)
            {
                checkCudaErrors(cudaFreeAsync(m_data, stream));
            }
            checkCudaErrors(cudaMallocAsync(&m_data, size, stream));
            m_capacity = size;
        }
        m_size = size;
    }

    auto size() const noexcept { return m_size; }
    auto data() const noexcept { return m_data; }
    auto capacity() const noexcept { return m_capacity; }

  private:
    size_t     m_size     = 0;
    size_t     m_capacity = 0;
    std::byte* m_data     = nullptr;
};
}  // namespace muda::details
