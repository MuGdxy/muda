#pragma once
#include <atomic>
#include <muda/muda_config.h>
namespace muda
{
class Debug
{
  private:
    static auto& _is_debug_sync_all()
    {
        static std::atomic<bool> m_is_debug_sync_all(false);
        return m_is_debug_sync_all;
    }

  public:
    static bool debug_sync_all(bool value)
    {
        _is_debug_sync_all() = value;
        return value;
    }

    static bool is_debug_sync_all() { return _is_debug_sync_all(); }
};
}  // namespace muda
