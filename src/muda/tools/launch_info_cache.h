#pragma once
#include <muda/tools/host_device_string_cache.h>
namespace muda::details
{
class LaunchInfoCache
{
  private:
    HostDeviceStringCache m_view_name_string_cache;
    HostDeviceStringCache m_kernel_name_string_cache;
    HostDeviceStringCache m_capture_name_string_cache;

    StringPointer m_current_kernel_name;
    StringPointer m_current_capture_name;

    LaunchInfoCache() MUDA_NOEXCEPT
    {
        m_current_kernel_name = m_kernel_name_string_cache[std::string_view{""}];
        m_current_capture_name = m_capture_name_string_cache[std::string_view{""}];
    }

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

    static auto current_capture_name(std::string_view name) MUDA_NOEXCEPT
    {
        auto& ins                  = instance();
        ins.m_current_capture_name = ins.m_capture_name_string_cache[name];
        return ins.m_current_capture_name;
    }

    static auto current_capture_name() MUDA_NOEXCEPT
    {
        return instance().m_current_capture_name;
    }

    static LaunchInfoCache& instance() MUDA_NOEXCEPT
    {
        thread_local static LaunchInfoCache instance;
        return instance;
    }
};
}  // namespace muda::details