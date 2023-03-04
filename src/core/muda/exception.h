#pragma once
#include <muda/muda_def.h>
#include <exception>
#include <string>

namespace muda
{
class exception : public std::exception
{
    std::string m_what;
  public:
    exception(const std::string& msg)
        : m_what(msg)
    {

    }
    virtual char const* what() const 
#ifdef MUDA_PLATFORM_LINUX
    MUDA_NOEXCEPT
#endif
    { return m_what.c_str(); }
};
}