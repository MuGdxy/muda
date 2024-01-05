#pragma once
#include <muda/viewer/viewer_base.h>

namespace muda
{

/*****************************************************************************
 *
 * Dense1D (1D array)
 * indexing (x)
 * 
 * A C/C++ array like viewer.
 * 
 *****************************************************************************/

template <typename T>
class Dense1DBase : public ViewerBase
{
  protected:
    T*  m_data;
    int m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC Dense1DBase() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC Dense1DBase(T* p, int dim) MUDA_NOEXCEPT : m_data(p), m_dim(dim)
    {
    }

    MUDA_GENERIC const T& operator()(int x) const MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x)];
    }

    MUDA_GENERIC int map(int x) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim))
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: out of range, index=(%d) m_dim=(%d)",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  m_dim);
        return x;
    }

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC int dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC Dense1DBase subview(int offset) const MUDA_NOEXCEPT
    {
        auto size = this->m_dim - offset;
        if constexpr(DEBUG_VIEWER)
        {
            if(offset < 0)
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: subview out of range, offset=%d size=%d m_dim=(%d)",
                                  this->name(),
                                  this->kernel_name(),
                                  offset,
                                  size,
                                  this->m_dim);
        }
        return Dense1DBase{this->m_data + offset, size};
    }

    MUDA_GENERIC Dense1DBase subview(int offset, int size) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
        {
            if(offset < 0 || offset + size > m_dim)
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: subview out of range, offset=%d size=%d m_dim=(%d)",
                                  this->name(),
                                  this->kernel_name(),
                                  offset,
                                  size,
                                  this->m_dim);
        }
        return Dense1DBase{this->m_data + offset, size};
    }

  protected:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("dense1D[%s:%s]: m_data is null",
                                  this->name(),
                                  this->kernel_name());
    }
};


template <typename T>
class CDense1D : public Dense1DBase<T>
{
    MUDA_VIEWER_COMMON_NAME(CDense1D);

  public:
    using Dense1DBase<T>::Dense1DBase;
    using Dense1DBase<T>::operator();
    MUDA_GENERIC CDense1D(const Dense1DBase<T>& base) MUDA_NOEXCEPT : Dense1DBase<T>(base)
    {
    }

    MUDA_GENERIC CDense1D(const T* p, int dim) MUDA_NOEXCEPT
        : Dense1DBase<T>(const_cast<T*>(p), dim)
    {
    }

    MUDA_GENERIC this_type subview(int offset) const MUDA_NOEXCEPT
    {
        return this_type{Dense1DBase<T>::subview(offset)};
    }

    MUDA_GENERIC this_type subview(int offset, int size) const MUDA_NOEXCEPT
    {
        return this_type{Dense1DBase<T>::subview(offset, size)};
    }
};


template <typename T>
class Dense1D : public Dense1DBase<T>
{
    MUDA_VIEWER_COMMON_NAME(Dense1D);

  public:
    using Dense1DBase<T>::Dense1DBase;
    using Dense1DBase<T>::operator();
    using Dense1DBase<T>::data;

    MUDA_GENERIC Dense1D(const Dense1DBase<T>& base) MUDA_NOEXCEPT : Dense1DBase<T>(base)
    {
    }

    MUDA_GENERIC operator CDense1D<T>() const MUDA_NOEXCEPT
    {
        return CDense1D<T>{*this};
    }

    MUDA_GENERIC T& operator()(int x) MUDA_NOEXCEPT
    {
        this->check();
        return this->m_data[this->map(x)];
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT { return this->m_data; }

    MUDA_GENERIC this_type subview(int offset) const MUDA_NOEXCEPT
    {
        return this_type{Dense1DBase<T>::subview(offset)};
    }

    MUDA_GENERIC this_type subview(int offset, int size) const MUDA_NOEXCEPT
    {
        return this_type{Dense1DBase<T>::subview(offset, size)};
    }
};

// viewer traits
template <typename T>
struct read_only_viewer<Dense1D<T>>
{
    using type = CDense1D<T>;
};

template <typename T>
struct read_write_viewer<CDense1D<T>>
{
    using type = Dense1D<T>;
};

// CTAD
template <typename T>
CDense1D(T*, int) -> CDense1D<T>;

template <typename T>
Dense1D(T*, int) -> Dense1D<T>;

// make functions
template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_cdense_1d(const T* data, int dimx) MUDA_NOEXCEPT
{
    return CDense1D<T>(data, dimx);
}

template <typename T, int N>
MUDA_INLINE MUDA_GENERIC auto make_cdense_1d(const T (&data)[N]) MUDA_NOEXCEPT
{
    return CDense1D<T>(data, N);
}

template <typename T>
MUDA_INLINE MUDA_GENERIC auto make_dense_1d(T* data, int dimx) MUDA_NOEXCEPT
{
    return Dense1D<T>(data, dimx);
}

template <typename T, int N>
MUDA_INLINE MUDA_GENERIC auto make_dense_1d(T (&data)[N]) MUDA_NOEXCEPT
{
    return Dense1D<T>(data, N);
}
}  // namespace muda