#pragma once
#include <muda/viewer/viewer_base.h>

namespace muda
{
template <typename T>
class DenseBase : public ViewerBase<false> // TODO
{
  protected:
    T* m_data;

  public:
    using value_type = T;

    MUDA_GENERIC DenseBase() MUDA_NOEXCEPT : m_data(nullptr) {}
    MUDA_GENERIC explicit DenseBase(T* p) MUDA_NOEXCEPT : m_data(p) {}

    MUDA_GENERIC DenseBase& operator=(const T& rhs) MUDA_NOEXCEPT
    {
        check();
        *m_data = rhs;
        return *this;
    }

    MUDA_GENERIC const T& operator()() const MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }

    MUDA_GENERIC const T& operator*() const MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }

    MUDA_GENERIC operator const T&() const MUDA_NOEXCEPT
    {
        check();
        return *m_data;
    }

    MUDA_GENERIC const T* operator->() const MUDA_NOEXCEPT
    {
        check();
        return m_data;
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

  protected:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
        {
            MUDA_KERNEL_ASSERT(m_data,
                               "dense[%s:%s]: m_data is null",
                               this->name(),
                               this->kernel_name());
        }
    }
};

template <typename T>
class CDense : public DenseBase<T>
{
    MUDA_VIEWER_COMMON_NAME(CDense);

  public:
    using DenseBase<T>::DenseBase;
    using DenseBase<T>::operator=;
    using DenseBase<T>::operator();

    MUDA_GENERIC CDense(const DenseBase<T>& base) MUDA_NOEXCEPT : DenseBase<T>(base)
    {
    }

    MUDA_GENERIC explicit CDense(const T* p) MUDA_NOEXCEPT
        : DenseBase<T>(const_cast<T*>(p))
    {
    }
};

template <typename T>
class Dense : public DenseBase<T>
{
    MUDA_VIEWER_COMMON_NAME(Dense);

  public:
    using DenseBase<T>::DenseBase;
    using DenseBase<T>::operator=;
    using DenseBase<T>::operator();
    using DenseBase<T>::data;

    MUDA_GENERIC Dense(const DenseBase<T>& base) MUDA_NOEXCEPT : DenseBase<T>(base)
    {
    }

    MUDA_GENERIC operator CDense<T>() const MUDA_NOEXCEPT
    {
        return CDense<T>{*this};
    }

    MUDA_GENERIC T& operator()() MUDA_NOEXCEPT
    {
        this->check();
        return *(this->m_data);
    }

    MUDA_GENERIC T& operator*() MUDA_NOEXCEPT
    {
        this->check();
        return *(this->m_data);
    }

    MUDA_GENERIC operator T&() MUDA_NOEXCEPT
    {
        this->check();
        return *(this->m_data);
    }

    MUDA_GENERIC T* operator->()
    {
        this->check();
        return this->m_data;
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT { return this->m_data; }
};

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


// CTAD
template <typename T>
CDense(const T*) -> CDense<T>;

template <typename T>
Dense(T*) -> Dense<T>;


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
