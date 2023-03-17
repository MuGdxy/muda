#pragma once
#include "base.h"

namespace muda
{
class DeviceSelect : public AlgBase<DeviceSelect>
{
  public:
    DeviceSelect(cudaStream_t stream = nullptr)
        : AlgBase(stream)
    {
    }

    //DeviceSelect& ExampleFunction(device_buffer<std::byte>& external_buffer, ...);
};
}  // namespace muda

#ifndef __INTELLISENSE__
#include "device_select.inl"
#endif

