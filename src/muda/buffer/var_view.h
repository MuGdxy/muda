#pragma once
#include <cuda.h>
#include <cinttypes>
#include <muda/viewer/dense.h>

namespace muda
{
template <typename T>
class VarViewBase
{
  protected:
    T* m_data = nullptr;

  public:
    MUDA_GENERIC VarViewBase() MUDA_NOEXCEPT : m_data(nullptr) {}
    MUDA_GENERIC VarViewBase(T* data) MUDA_NOEXCEPT : m_data(data) {}

    MUDA_GENERIC const T* data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC CDense<T> cviewer() const MUDA_NOEXCEPT;
};

template <typename T>
class CVarView : public VarViewBase<T>
{
  public:
    using VarViewBase::VarViewBase;

    MUDA_GENERIC CVarView(const VarViewBase<T>& base) MUDA_NOEXCEPT : VarViewBase<T>(base)
    {
    }

    MUDA_GENERIC CVarView(const T* data) MUDA_NOEXCEPT
        : VarViewBase<T>(const_cast<T*>(data))
    {
    }

    void copy_to(T* data) const;
};

template <typename T>
class VarView : public VarViewBase<T>
{
  public:
    using VarViewBase::data;
    using VarViewBase::VarViewBase;

    MUDA_GENERIC VarView(VarViewBase<T> base) MUDA_NOEXCEPT : VarViewBase<T>(base)
    {
    }

    MUDA_GENERIC operator CVarView<T>() const MUDA_NOEXCEPT
    {
        return CVarView<T>{*this};
    }

    MUDA_GENERIC T* data() MUDA_NOEXCEPT { return this->m_data; }

    void copy_from(const T* data);
    void copy_from(CVarView<T> data);
    void fill(const T& value);

    MUDA_GENERIC Dense<T> viewer() MUDA_NOEXCEPT;
};

// viewer traits
template <typename T>
struct read_only_viewer<VarView<T>>
{
    using type = CVarView<T>;
};

template <typename T>
struct read_write_viewer<CVarView<T>>
{
    using type = VarView<T>;
};

// CTAD
template <typename T>
CVarView(T*) -> CVarView<T>;

template <typename T>
VarView(T*) -> VarView<T>;
}  // namespace muda


#include "details/var_view.inl"