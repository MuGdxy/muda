#pragma once
#include <muda/viewer/viewer_base.h>
#include <muda/viewer/dense.h>
#include <muda/viewer/details/cse_check.inl>

namespace muda
{
template <typename T>
class CCSEViewer : public RWViewer
{
    MUDA_VIEWER_COMMON_NAME(CCSEViewer);
    const T* m_data;

    int* m_begin;
    int* m_count;

    int m_ndata;
    int m_dim_i;

  public:
    MUDA_GENERIC CCSEViewer() MUDA_NOEXCEPT : m_data(nullptr),
                                              m_begin(nullptr),
                                              m_count(nullptr),
                                              m_ndata(0),
                                              m_dim_i(0)
    {
    }

    MUDA_GENERIC CCSEViewer(const T* data, int ndata, int* begin, int* count, int dim_i) MUDA_NOEXCEPT
        : m_data(data),
          m_begin(begin),
          m_count(count),
          m_ndata(ndata),
          m_dim_i(dim_i)
    {
    }

    MUDA_GENERIC const T& operator()(int i, int j) const MUDA_NOEXCEPT
    {
        return m_data[cal_global_offset(i, j)];
    }

    MUDA_GENERIC int dim_i() const MUDA_NOEXCEPT { return m_dim_i; }

    MUDA_GENERIC int dim_j(int i) const MUDA_NOEXCEPT
    {
        check_dimi(i);
        return m_count[i];
    }

    MUDA_GENERIC int ndata(int i) const MUDA_NOEXCEPT { return m_ndata; }

    // get the i-th row of the sparse 2d data structure
    MUDA_GENERIC CDense1D<T> operator()(int i) MUDA_NOEXCEPT
    {
        check_dimi(i);
        return CDense1D<T>(m_data + m_begin[i], m_count[i]);
    }

  private:
    MUDA_INLINE MUDA_GENERIC void check_data() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::cse_check_data(m_data, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_dimi(int i) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::cse_check_dimi(i, m_dim_i, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_dimj(int i, int j, int dimj) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::cse_check_dimj(i, j, dimj, m_dim_i, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_global_offset(int i, int j, int dimj, int global_offset) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::cse_check_global_offset(
                i, j, dimj, global_offset, m_dim_i, m_ndata, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC int cal_global_offset(int i, int j) const MUDA_NOEXCEPT
    {
        check_dimi(i);
        auto dimj = m_count[i];
        check_dimj(i, j, dimj);
        int global_offset = m_begin[i] + j;
        check_global_offset(i, j, dimj, global_offset);
        return global_offset;
    }
};


/// <summary>
/// compressed sparse element viewer.
/// using:
///     a 1D [data] array of type T.
///     a 1D [begin] index array of type int.
///     a 1D [count] array of type int.
/// cse(i,j) -> data[begin[i] + j]
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class CSEViewer : public RWViewer
{
    T* m_data;

    int* m_begin;
    int* m_count;

    int m_ndata;
    int m_dim_i;

  public:
    MUDA_GENERIC CSEViewer() MUDA_NOEXCEPT : m_data(nullptr),
                                             m_begin(nullptr),
                                             m_count(nullptr),
                                             m_ndata(0),
                                             m_dim_i(0)
    {
    }

    MUDA_GENERIC CSEViewer(T* data, int ndata, int* begin, int* count, int dim_i) MUDA_NOEXCEPT
        : m_data(data),
          m_begin(begin),
          m_count(count),
          m_ndata(ndata),
          m_dim_i(dim_i)
    {
    }

    MUDA_GENERIC const T& operator()(int i, int j) const MUDA_NOEXCEPT
    {
        return m_data[cal_global_offset(i, j)];
    }

    MUDA_GENERIC T& operator()(int i, int j) MUDA_NOEXCEPT
    {
        return m_data[cal_global_offset(i, j)];
    }

    MUDA_GENERIC int dim_i() const MUDA_NOEXCEPT { return m_dim_i; }

    MUDA_GENERIC int dim_j(int i) const MUDA_NOEXCEPT
    {
        check_dimi(i);
        return m_count[i];
    }

    MUDA_GENERIC int ndata(int i) const MUDA_NOEXCEPT { return m_ndata; }

    // get the i-th row of the sparse 2d data structure
    MUDA_GENERIC Dense1D<T> operator()(int i) MUDA_NOEXCEPT
    {
        check_dimi(i);
        return Dense1D<T>(m_data + m_begin[i], m_count[i]);
    }

  private:
    MUDA_INLINE MUDA_GENERIC void check_data() const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::cse_check_data(m_data, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_dimi(int i) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::cse_check_dimi(i, m_dim_i, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_dimj(int i, int j, int dimj) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::cse_check_dimj(i, j, dimj, m_dim_i, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_global_offset(int i, int j, int dimj, int global_offset) const MUDA_NOEXCEPT
    {
        if constexpr(DEBUG_VIEWER)
            details::cse_check_global_offset(
                i, j, dimj, global_offset, m_dim_i, m_ndata, this->name(), this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC int cal_global_offset(int i, int j) const MUDA_NOEXCEPT
    {
        check_dimi(i);
        auto dimj = m_count[i];
        check_dimj(i, j, dimj);
        int global_offset = m_begin[i] + j;
        check_global_offset(i, j, dimj, global_offset);
        return global_offset;
    }
};

template <typename T>
struct read_only_view<CSEViewer<T>>
{
    using type = CCSEViewer<T>;
};

template <typename T>
struct read_write_view<CCSEViewer<T>>
{
    using type = CSEViewer<T>;
};
}  // namespace muda