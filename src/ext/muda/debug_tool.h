#pragma once
#include <muda/thread_only/mstring.h>

namespace muda::debug
{
using string = thread_only::string;

template <typename... Args>
inline __device__ void log(const string& fmt, Args&&... args)
{
	// TODO:
}

}  // namespace muda::debug
