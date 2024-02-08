#pragma once
#include <muda/tools/launch_info_cache.h>
#include <string_view>
namespace muda
{
class KernelLabel
{
  public:
    KernelLabel(std::string_view name)
    {
        if constexpr(muda::RUNTIME_CHECK_ON)
            details::LaunchInfoCache::current_kernel_name(name);
    }

    ~KernelLabel()
    {
        if constexpr(muda::RUNTIME_CHECK_ON)
            details::LaunchInfoCache::current_kernel_name("");
    }
};
}  // namespace muda
