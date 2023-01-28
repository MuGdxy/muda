#pragma once
#include "EASTL/string.h"
#include "thread_allocator.h"
namespace muda::thread_only
{
using string = eastl::string;
}  // namespace muda::thread_only

namespace muda
{
inline MUDA_GENERIC const char* printConvert(const thread_only::string& s)
{
    return s.c_str();
}
}  // namespace muda