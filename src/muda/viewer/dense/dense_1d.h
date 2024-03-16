/*****************************************************************/ /**
 * \file   dense_1d.h
 * \brief  A C/C++ array like viewer for kernel access, with safe checking
 * on any input. You can index the element in `Dense1D<T>` by `operator ()`.
 * 
 * \author MuGdxy
 * \date   January 2024
 *********************************************************************/

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

template <bool IsConst, typename T>
class Dense1DBase : public ViewerBase<IsConst>
{
    using Base = ViewerBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

    MUDA_VIEWER_COMMON_NAME(Dense1DBase);

  public:
    using ConstViewer    = Dense1DBase<true, T>;
    using NonConstViewer = Dense1DBase<false, T>;
    using ThisViewer     = Dense1DBase<IsConst, T>;

  protected:
    auto_const_t<T>* m_data;
    int              m_dim;

  public:
    using value_type = T;

    MUDA_GENERIC Dense1DBase() MUDA_NOEXCEPT : m_data(nullptr) {}

    MUDA_GENERIC Dense1DBase(auto_const_t<T>* p, int dim) MUDA_NOEXCEPT : m_data(p),
                                                                          m_dim(dim)
    {
    }

    MUDA_GENERIC auto as_const() const MUDA_NOEXCEPT
    {
        return ConstViewer{m_data, m_dim};
    }

    MUDA_GENERIC operator ConstViewer() const MUDA_NOEXCEPT
    {
        return as_const();
    }


    MUDA_GENERIC auto_const_t<T>& operator()(int x) MUDA_NOEXCEPT
    {
        check();
        return m_data[map(x)];
    }

    MUDA_GENERIC const T& operator()(int x) const MUDA_NOEXCEPT
    {
        return remove_const(*this)(x);
    }

    MUDA_GENERIC auto_const_t<T>* data() MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC const T*         data() const MUDA_NOEXCEPT { return m_data; }


    MUDA_GENERIC int total_size() const MUDA_NOEXCEPT { return m_dim; }
    MUDA_GENERIC int dim() const MUDA_NOEXCEPT { return m_dim; }

    MUDA_GENERIC ThisViewer subview(int offset) MUDA_NOEXCEPT
    {
        auto size = this->m_dim - offset;
        if constexpr(DEBUG_VIEWER)
        {
            if(offset < 0)
                MUDA_KERNEL_ERROR("Dense1D[%s:%s]: subview out of range, offset=%d size=%d m_dim=(%d)",
                                  this->name(),
                                  this->kernel_name(),
                                  offset,
                                  size,
                                  this->m_dim);
        }
        return ThisViewer{this->m_data + offset, size};
    }

    MUDA_GENERIC ThisViewer subview(int offset, int size) MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
        {
            if(offset < 0 || offset + size > m_dim)
                MUDA_KERNEL_ERROR("Dense1D[%s:%s]: subview out of range, offset=%d size=%d m_dim=(%d)",
                                  this->name(),
                                  this->kernel_name(),
                                  offset,
                                  size,
                                  this->m_dim);
        }
        return ThisViewer{this->m_data + offset, size};
    }

    MUDA_GENERIC ConstViewer subview(int offset) const MUDA_NOEXCEPT
    {
        return remove_const(*this).subview(offset).as_const();
    }

    MUDA_GENERIC ConstViewer subview(int offset, int size) const MUDA_NOEXCEPT
    {
        return remove_const(*this).subview(offset, size).as_const();
    }

  protected:
    MUDA_INLINE MUDA_GENERIC void check() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(m_data == nullptr)
                MUDA_KERNEL_ERROR("Dense1D[%s:%s]: m_data is null",
                                  this->name(),
                                  this->kernel_name());
    }

    MUDA_GENERIC int map(int x) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            if(!(x >= 0 && x < m_dim))
                MUDA_KERNEL_ERROR("Dense1D[%s:%s]: out of range, index=(%d) m_dim=(%d)",
                                  this->name(),
                                  this->kernel_name(),
                                  x,
                                  m_dim);
        return x;
    }
};

template <typename T>
using Dense1D = Dense1DBase<false, T>;

template <typename T>
using CDense1D = Dense1DBase<true, T>;

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