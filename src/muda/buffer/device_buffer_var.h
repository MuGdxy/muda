#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <optional>
#include <vector>
#include <muda/container/vector.h>
#include <muda/container/var.h>
#include <muda/launch/launch_base.h>
#include <muda/launch/memory.h>
#include <muda/launch/parallel_for.h>

namespace muda
{
template <typename T>
class DeviceBufferVar
{
  private:
    T* m_data;

  public:
    using value_type = T;

    DeviceBufferVar();

    DeviceBufferVar(const DeviceBufferVar& other) { copy_from(other).wait(); }

    DeviceBufferVar(DeviceBufferVar&& other) MUDA_NOEXCEPT : m_data(other.m_data)
    {
        other.m_data = nullptr;
    }

    DeviceBufferVar& operator=(const DeviceBufferVar<value_type>& other);
    DeviceBufferVar& operator=(const DeviceVar<value_type>& other);
    DeviceBufferVar& operator=(const value_type& other);

    Empty copy_from(const value_type& var);
    Empty copy_to(value_type& var) const;

    Empty copy_from(const DeviceVar<value_type>& var);
    Empty copy_to(DeviceVar<value_type>& var) const;

    Empty copy_from(const DeviceBufferVar<value_type>& var);
    Empty copy_to(DeviceBufferVar<value_type>& var) const;

    operator T() const
    {
        T val;
        copy_to(val).wait();
        return val;
    }

    T*       data() { return m_data; }
    const T* data() const { return m_data; }
    bool     already_init() const { return m_init; }

    Dense<T>  viewer();
    CDense<T> cviewer() const;
};

#include "details/device_buffer_var.inl"