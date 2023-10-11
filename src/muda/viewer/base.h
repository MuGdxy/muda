#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <muda/muda_def.h>
#include <muda/tools/debug_log.h>
#include <muda/debug.h>
#include <muda/muda_config.h>
#include <muda/assert.h>
#include <muda/tools/launch_info_cache.h>
#include <muda/tools/fuzzy.h>
#include <muda/type_traits/type_modifier.h>

#define $def(viewer, from) viewer = (from).name(#viewer)

namespace muda
{
class ViewerBase
{
#if MUDA_CHECK_ON
    details::StringPointer m_view_name;
    details::StringPointer m_kernel_name;
#endif
  public:
    MUDA_GENERIC ViewerBase()
    {
#if MUDA_CHECK_ON
#ifndef __CUDA_ARCH__
        m_kernel_name = details::LaunchInfoCache::current_kernel_name();
#endif
#endif
    }

    MUDA_GENERIC const char* name() const MUDA_NOEXCEPT
    {
#if MUDA_CHECK_ON

        auto n = m_view_name.auto_select();
        if(n)
            return n;
#endif
        return "~";
    }

    MUDA_GENERIC const char* kernel_name() const MUDA_NOEXCEPT
    {
#if MUDA_CHECK_ON
        auto n = m_kernel_name.auto_select();
        if(n)
            return n;
#endif
        return "~";
    }

    // default copy / move
    ViewerBase(const ViewerBase&)            = default;
    ViewerBase(ViewerBase&&)                 = default;
    ViewerBase& operator=(const ViewerBase&) = default;
    ViewerBase& operator=(ViewerBase&&)      = default;

  protected:
    MUDA_INLINE MUDA_HOST void name(const char* n) MUDA_NOEXCEPT
    {
#if MUDA_CHECK_ON
        m_view_name = details::LaunchInfoCache::view_name(n);
#endif
    }
};

// Read Write Viewer
class RWViewer : public ViewerBase
{
};

// Read Only Viewer
class ROViewer : public ViewerBase
{
};

#define MUDA_VIEWER_COMMON(viewer_name)                                        \
  public:                                                                      \
    using this_type = viewer_name;                                             \
                                                                               \
    MUDA_INLINE MUDA_HOST this_type& name(const char* n) noexcept              \
    {                                                                          \
        ViewerBase::name(n);                                                   \
        return *this;                                                          \
    }                                                                          \
                                                                               \
    MUDA_INLINE MUDA_GENERIC const char* name() const noexcept                 \
    {                                                                          \
        return ViewerBase::name();                                             \
    }                                                                          \
                                                                               \
  private:


}  // namespace muda