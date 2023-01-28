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
class viewer_base
{
    char m_name[VIEWER_NAME_MAX];

  public:
    MUDA_GENERIC const char* name() const noexcept
    {
        if constexpr(VIEWER_NAME_MAX > 0)
            return m_name;
        else
            return "unnamed";
    }
    MUDA_GENERIC Derived& name(const char* n) noexcept
    {
        if constexpr(VIEWER_NAME_MAX > 0)
        {
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
        }
        return derived();
    }

  private:
    MUDA_GENERIC Derived& derived() { return (Derived&)(*this); }
};
}  // namespace muda