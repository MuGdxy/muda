#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <Eigen/Core>
#include "../muda_def.h"
#include "../tools/debug_log.h"
#include "../muda_config.h"
#include "../assert.h"

namespace muda
{
template <typename Derived>
class ViewBase
{
#if !MUDA_CHECK_ON
    // msvc doesn't allow 0 size array in base class
    char m_name[VIEWER_NAME_MAX];
#endif
  public:
    ViewBase()
    {
#if !MUDA_CHECK_ON
        m_name[0] = '\0';
#endif
    }
    
    MUDA_GENERIC const char* name() const MUDA_NOEXCEPT
    {
#if !MUDA_CHECK_ON
        if(m_name[0] != '\0')
            return m_name;
#endif
        return "unnamed";
    }

    MUDA_GENERIC Derived& name(const char* n) MUDA_NOEXCEPT
    {
#if !MUDA_CHECK_ON
        if(n == nullptr)
        {
            m_name[0] = '\0';
            return derived();
        }

        int  i      = 0;
        bool finish = false;
        for(; i < VIEWER_NAME_MAX; ++i)
        {
            auto c    = n[i];
            m_name[i] = c;
            if(c == '\0')
            {
                finish = true;
                break;
            }
        }

        if(!finish)
        {
            m_name[VIEWER_NAME_MAX - 1] = '\0';
            muda_kernel_warn("viewer name [%s] is too long, truncated to [%s]\n", n, name());
        }
#endif
        return derived();
    }

  private:
    MUDA_GENERIC Derived& derived() { return (Derived&)(*this); }
};
}  // namespace muda