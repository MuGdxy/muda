#pragma once
#include <muda/viewer/viewer_base.h>

namespace muda
{
template <bool IsConst, typename T>
class DenseViewerBase : public ViewerBase<IsConst>
{
    using Base = ViewerBase<IsConst>;
    MUDA_VIEWER_COMMON_NAME(DenseViewerBase);

  public:
    using ConstViewer    = DenseViewerBase<true, T>;
    using NonConstViewer = DenseViewerBase<false, T>;
    using ThisViewer     = DenseViewerBase<IsConst, T>;

  protected:
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;
    auto_const_t<T>* m_data;

  public:
    using value_type = T;

    MUDA_GENERIC DenseViewerBase() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC explicit DenseViewerBase(auto_const_t<T>* p) MUDA_NOEXCEPT : m_data(p)
    {
    }

    MUDA_GENERIC auto as_const() const MUDA_NOEXCEPT
    {
        return ConstViewer{m_data};
    }

    MUDA_GENERIC operator ConstViewer() const MUDA_NOEXCEPT
    {
        return as_const();
    }

    MUDA_GENERIC auto_const_t<T>& operator*() MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }

    MUDA_GENERIC auto_const_t<T>* operator->() MUDA_NOEXCEPT
    {
        check();
        return m_data;
    }

    MUDA_GENERIC auto_const_t<T>* data() MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC const T& operator*() const MUDA_NOEXCEPT
    {
        return remove_const(*this).operator*();
    }

    MUDA_GENERIC const T* operator->() const MUDA_NOEXCEPT
    {
        return remove_const(*this).operator->();
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC operator const T&() const MUDA_NOEXCEPT { return *m_data; }

  protected:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
        {
            MUDA_KERNEL_ASSERT(m_data,
                               "Dense[%s:%s]: m_data is null",
                               this->name(),
                               this->kernel_name());
        }
    }
};

template <typename T>
class Dense : public DenseViewerBase<false, T>
{
    MUDA_VIEWER_COMMON_NAME(Dense);

  public:
    using Base           = DenseViewerBase<false, T>;
    using ConstViewer    = typename Base::ConstViewer;
    using NonConstViewer = Dense<T>;
    using ThisViewer     = Dense<T>;

    using Base::Base;

    MUDA_GENERIC Dense(const Base& base) MUDA_NOEXCEPT : Base(base) {}

    MUDA_GENERIC Dense& operator=(const T& base) MUDA_NOEXCEPT
    {
        Base::check();
        *this->m_data = base;
        return *this;
    }

    using Base::operator*;
    using Base::operator->;
    using Base::as_const;
    using Base::data;

    MUDA_GENERIC operator T&() MUDA_NOEXCEPT { return *this->m_data; }
    MUDA_GENERIC operator const T&() const MUDA_NOEXCEPT
    {
        return *this->m_data;
    }

    MUDA_GENERIC operator ConstViewer() const MUDA_NOEXCEPT
    {
        return this->as_const();
    }
};

template <typename T>
using CDense = DenseViewerBase<true, T>;

// viewer traits
template <typename T>
struct read_only_viewer<Dense<T>>
{
    using type = CDense<T>;
};

template <typename T>
struct read_write_viewer<CDense<T>>
{
    using type = Dense<T>;
};

// make functions
template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense(const T* data) MUDA_NOEXCEPT
{
    return CDense<T>{data};
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense(T* data) MUDA_NOEXCEPT
{
    return Dense<T>{data};
}

//print convert
template <typename T>
MUDA_INLINE MUDA_GENERIC const T& print_convert(const Dense<T>& v) MUDA_NOEXCEPT
{
    return v.operator const T&();
}

template <typename T>
MUDA_INLINE MUDA_GENERIC const T& print_convert(const CDense<T>& v) MUDA_NOEXCEPT
{
    return v.operator const T&();
}

}  // namespace muda
