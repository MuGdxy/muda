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
    VarViewBase() MUDA_NOEXCEPT : m_data(nullptr) {}
    VarViewBase(T* data) MUDA_NOEXCEPT : m_data(data) {}

    const T* data() const MUDA_NOEXCEPT { return m_data; }

    CDense<T> cviewer() const MUDA_NOEXCEPT;
};

template <typename T>
class CVarView : public VarViewBase<T>
{
  public:
    using VarViewBase::VarViewBase;

    CVarView(const VarViewBase<T>& base) MUDA_NOEXCEPT : VarViewBase<T>(base) {}

    CVarView(const T* data) MUDA_NOEXCEPT : VarViewBase<T>(const_cast<T*>(data))
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

    VarView(VarViewBase<T> base) MUDA_NOEXCEPT : VarViewBase<T>(base) {}

    operator CVarView<T>() const MUDA_NOEXCEPT { return CVarView<T>{*this}; }

    T* data() MUDA_NOEXCEPT { return this->m_data; }

    void copy_from(const T* data);
    void copy_from(CVarView<T> data);
    void fill(const T& value);

    Dense<T> viewer() MUDA_NOEXCEPT;
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