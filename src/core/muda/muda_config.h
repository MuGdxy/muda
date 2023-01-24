#pragma once
#ifndef MUDA_NDEBUG
#define MUDA_NDEBUG 0
#endif
namespace muda
{
constexpr bool NO_CHECK = MUDA_NDEBUG;
namespace config
{
    constexpr bool on(bool cond = false)
    {
        return cond && !NO_CHECK;
    }
}  // namespace config
// debug viewer
constexpr bool DEBUG_VIEWER = config::on(true);
// debug ticcd
constexpr bool DEBUG_TICCD = config::on(true);
// debug thread only container
constexpr bool DEBUG_THREAD_ONLY = config::on(true);
// debug container
constexpr bool DEBUG_CONTAINER = config::on(true);
// debug composite
constexpr bool DEBUG_COMPOSITE = config::on(true);
// trap on error happens
constexpr bool TRAP_ON_ERROR = config::on(true);
// light workload block size
constexpr int LIGHT_WORKLOAD_BLOCK_SIZE = 256;
// middle workload block size
constexpr int MIDDLE_WORKLOAD_BLOCK_SIZE = 128;
// heavy workload block size
constexpr int HEAVY_WORKLOAD_BLOCK_SIZE = 64;
// view name max length
constexpr int VIEWER_NAME_MAX = MUDA_NDEBUG ? 0 : 16;
}  // namespace muda


#define EASTL_ASSERT_ENABLED 1
#define EASTL_EMPTY_REFERENCE_ASSERT_ENABLED 1
