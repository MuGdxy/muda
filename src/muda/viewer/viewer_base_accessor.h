#pragma once
#include <muda/viewer/viewer_base.h>

namespace muda
{
namespace details
{
    class ViewerBaseAccessor
    {
      public:
        template <bool IsConst>
        MUDA_GENERIC static auto& kernel_name(ViewerBase<IsConst>& viewer)
        {
            return viewer.m_kernel_name;
        }
        template <bool IsConst>
        MUDA_GENERIC static const auto& kernel_name(const ViewerBase<IsConst>& viewer)
        {
            return viewer.m_kernel_name;
        }
        template <bool IsConst>
        MUDA_GENERIC static auto& viewer_name(ViewerBase<IsConst>& viewer)
        {
            return viewer.m_viewer_name;
        }
        template <bool IsConst>
        MUDA_GENERIC static const auto& viewer_name(const ViewerBase<IsConst>& viewer)
        {
            return viewer.m_viewer_name;
        }
    };
}  // namespace details
}  // namespace muda