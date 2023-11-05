#pragma once
#include <muda/viewer/viewer_base.h>

namespace muda
{
namespace details
{
    class ViewerBaseAccessor
    {
      public:
        MUDA_GENERIC static auto& kernel_name(ViewerBase& viewer)
        {
            return viewer.m_kernel_name;
        }
        MUDA_GENERIC static auto& viewer_name(ViewerBase& viewer)
        {
            return viewer.m_viewer_name;
        }
    };
}  // namespace details
}  // namespace muda