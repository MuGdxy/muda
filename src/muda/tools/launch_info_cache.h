#pragma once
#include <muda/tools/host_device_string_cache.h>
namespace muda::details
{
#if MUDA_CHECK_ON
class LaunchInfoCache
{
  private:
    HostDeviceStringCache m_view_name_string_cache;
    HostDeviceStringCache m_kernel_name_string_cache;
    StringPointer         m_current_kernel_name;
<<<<<<< HEAD
    LaunchInfoCache() MUDA_NOEXCEPT {}
=======
    LaunchInfoCache() MUDA_NOEXCEPT
    {
        m_view_name_string_cache[""];
        m_kernel_name_string_cache[""];
    }
>>>>>>> a42cfbff779b40bd2e14d6d04f3931baf8843a7b

  public:
    static auto view_name(std::string_view name) MUDA_NOEXCEPT
    {
        return instance().m_view_name_string_cache[name];
    }
    static auto current_kernel_name(std::string_view name) MUDA_NOEXCEPT
    {
        auto& ins                 = instance();
        ins.m_current_kernel_name = ins.m_kernel_name_string_cache[name];
        return ins.m_current_kernel_name;
    }

    static auto current_kernel_name() MUDA_NOEXCEPT
    {
        return instance().m_current_kernel_name;
    }

    static LaunchInfoCache& instance() MUDA_NOEXCEPT
    {
        thread_local static LaunchInfoCache instance;
        return instance;
    }
};
#endif
}  // namespace muda::details