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

    //static auto& _retrieve_on_wait()
    //{
    //    static std::atomic<bool> m_retrieve_on_wait(false);
    //    return m_retrieve_on_wait;
    //}

  public:
    static bool debug_sync_all(bool value)
    {
        _is_debug_sync_all() = value;
        return value;
    }

    //static bool retrieve_on_wait(bool value)
    //{
    //    _retrieve_on_wait() = value;
    //    return value;
    //}

    static bool is_debug_sync_all() { return _is_debug_sync_all(); }
    //static bool is_retrieve_on_wait() { return _retrieve_on_wait(); }

    //static void init_logger();
    //static void logger_retrieve(std::ostream& os);
};
}  // namespace muda


//#include <muda/logger/logger.h>
//#include <iostream>
//namespace muda
//{
//MUDA_INLINE void Debug::init_logger()
//{
//    auto guard = std::lock_guard{Logger::mutex()};
//    Logger::instance();
//    retrieve_on_wait(true);
//}
//
//MUDA_INLINE void Debug::logger_retrieve(std::ostream& os = std::cout)
//{
//    auto guard = std::lock_guard{Logger::mutex()};
//    Logger::instance().retrieve(os);
//}
//}  // namespace muda
