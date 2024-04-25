#pragma once
#include <cuda.h>
#include <cinttypes>
#include <muda/view/view_base.h>
#include <muda/viewer/dense.h>

namespace muda
{
template <bool IsConst, typename T>
class VarViewBase : public ViewBase<IsConst>
{
    using Base = ViewBase<IsConst>;

  protected:
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

    auto_const_t<T>* m_data = nullptr;

  public:
    using ConstView    = VarViewBase<true, T>;
    using NonConstView = VarViewBase<false, T>;
    using ThisView     = VarViewBase<IsConst, T>;

    using ConstViewer    = CDense<T>;
    using NonConstViewer = Dense<T>;
    using ThisViewer = typename std::conditional_t<IsConst, ConstViewer, NonConstViewer>;


    MUDA_GENERIC VarViewBase() MUDA_NOEXCEPT : m_data(nullptr) {}
    MUDA_GENERIC VarViewBase(auto_const_t<T>* data) MUDA_NOEXCEPT : m_data(data)
    {
    }

    MUDA_GENERIC auto_const_t<T>* data() MUDA_NOEXCEPT { return m_data; }
    MUDA_GENERIC const T*         data() const MUDA_NOEXCEPT { return m_data; }

    MUDA_GENERIC auto cviewer() const MUDA_NOEXCEPT
    {
        return ConstViewer{m_data};
    }
    MUDA_GENERIC auto viewer() MUDA_NOEXCEPT { return ThisViewer{m_data}; }

    /**********************************************************************************
    * VarView As Iterator
    ***********************************************************************************/

    // Random Access Iterator Interface
    using value_type        = T;
    using reference         = T&;
    using pointer           = T*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = size_t;

    MUDA_GENERIC reference operator*() { return *data(); }
    MUDA_GENERIC auto_const_t<T>& operator[](int i) { return *data(); }
    MUDA_GENERIC const T&         operator[](int i) const { return *data(); }
};

template <typename T>
class CVarView : public VarViewBase<true, T>
{
    using Base = VarViewBase<true, T>;

  public:
    using Base::Base;
    MUDA_GENERIC CVarView(const Base& base) MUDA_NOEXCEPT : Base(base){};
    void         copy_to(T* data) const;
};

template <typename T>
class VarView : public VarViewBase<false, T>
{
    using Base = VarViewBase<false, T>;

  public:
    using Base::Base;
    MUDA_GENERIC VarView(const Base& base) MUDA_NOEXCEPT : Base(base) {}

    MUDA_GENERIC auto as_const() const MUDA_NOEXCEPT
    {
        return CVarView<T>{this->m_data};
    }

    void copy_from(const T* data);
    void copy_from(CVarView<T> data);
    void fill(const T& value);
    void copy_to(T* data) const { as_const().copy_to(data); }
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