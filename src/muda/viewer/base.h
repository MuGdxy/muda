#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <Eigen/Core>
#include <muda/muda_def.h>
#include <muda/tools/debug_log.h>
#include <muda/muda_config.h>
#include <muda/assert.h>

namespace muda
{
class ViewerBase
{
#if MUDA_CHECK_ON
    // msvc doesn't allow 0 size array in base class
    char m_name[VIEWER_NAME_MAX];
#endif
  public:
    ViewerBase()
    {
#if MUDA_CHECK_ON
        m_name[0] = '\0';
#endif
    }

    MUDA_GENERIC const char* name() const MUDA_NOEXCEPT
    {
#if MUDA_CHECK_ON
        if(m_name[0] != '\0')
            return m_name;
#endif
        return "";
    }

    // copy ctor
    MUDA_GENERIC ViewerBase(const ViewerBase& rhs) MUDA_NOEXCEPT
    {
#if MUDA_CHECK_ON
        copy_name(rhs.m_name);
#endif
    }

    // copy assignment
    MUDA_GENERIC ViewerBase& operator=(const ViewerBase& rhs) MUDA_NOEXCEPT
    {
#if MUDA_CHECK_ON
        if(this == &rhs)
            return *this;
        copy_name(rhs.m_name);
#endif
        return *this;
    }

    // move ctor
    MUDA_GENERIC ViewerBase(ViewerBase&& rhs) MUDA_NOEXCEPT
    {
#if MUDA_CHECK_ON
        copy_name(rhs.m_name);
        rhs.m_name[0] = '\0';
#endif
    }

    // move assignment
    MUDA_GENERIC ViewerBase& operator=(ViewerBase&& rhs) MUDA_NOEXCEPT
    {
#if MUDA_CHECK_ON
        if(this == &rhs)
            return *this;
        copy_name(rhs.m_name);
        rhs.m_name[0] = '\0';
#endif
        return *this;
    }


  protected:
    MUDA_GENERIC void name(const char* n) MUDA_NOEXCEPT
    {
#if MUDA_CHECK_ON
        if(n == nullptr)
        {
            m_name[0] = '\0';
            return;
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
    }

  private:
#if MUDA_CHECK_ON
    MUDA_GENERIC void copy_name(const char* n) MUDA_NOEXCEPT
    {
        for(int i = 0; i < VIEWER_NAME_MAX; ++i)
            this->m_name[i] = n[i];
    }
#endif
};

// Read Write Viewer
class RWViewer : public ViewerBase
{
};

// Read Only Viewer
class ROViewer : public ViewerBase
{
};


//#define MUDA_VIEWER_COMMON(viewer_name)                                        \
//  public:                                                                      \
//    using this_type = viewer_name;                                             \
//    this_type&& name(const char* n)&& noexcept                                 \
//    {                                                                          \
//        std::move(*this)._set_viewer_name(n);                                  \
//        return std::move(*this);                                               \
//    }                                                                          \
//    this_type& name(const char* n) noexcept                                    \
//    {                                                                          \
//        (*this)._set_viewer_name(n);                                           \
//        return *this;                                                          \
//    }                                                                          \
//                                                                               \
//  private:

#define MUDA_VIEWER_COMMON(viewer_name)                                        \
  public:                                                                      \
    using this_type = viewer_name;                                             \
                                                                               \
    MUDA_INLINE MUDA_GENERIC this_type& name(const char* n) noexcept           \
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