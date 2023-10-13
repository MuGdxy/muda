#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <muda/viewer/dense.h>
namespace muda
{
template <typename T>
class DeviceVar;

template <typename T>
class DeviceBufferVar
{
  private:
    T* m_data;

  public:
    using value_type = T;

    DeviceBufferVar();

    DeviceBufferVar(const DeviceBufferVar& other);

    DeviceBufferVar(DeviceBufferVar&& other) MUDA_NOEXCEPT;

    DeviceBufferVar& operator=(const DeviceBufferVar<value_type>& other);
    DeviceBufferVar& operator=(const DeviceVar<value_type>& other);

    // copy from host
    DeviceBufferVar& operator=(const value_type& val);
    // copy to host
    operator T() const;

    T*       data() { return m_data; }
    const T* data() const { return m_data; }

    Dense<T>  viewer();
    CDense<T> cviewer() const;
};
}  // namespace muda

#include "details/device_buffer_var.inl"