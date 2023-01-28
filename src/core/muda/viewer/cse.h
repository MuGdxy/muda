#pragma once
#include "base.h"
#include "dense.h"

namespace muda
{
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
class cse : public viewer_base<cse<T>>
{
    T* m_data;

    int* m_begin;
    int* m_count;

    int m_ndata;
    int m_dim_i;

  public:
    MUDA_GENERIC cse() noexcept
        : m_data(nullptr)
        , m_begin(nullptr)
        , m_count(nullptr)
        , m_ndata(0)
        , m_dim_i(0)
    {
    }

    MUDA_GENERIC cse(T* data, int ndata, int* begin, int* count, int dim_i) noexcept
        : m_data(data)
        , m_begin(begin)
        , m_count(count)
        , m_ndata(ndata)
        , m_dim_i(dim_i)
    {
    }

    MUDA_GENERIC const T& operator()(int i, int j) const noexcept
    {
        return m_data[cal_global_offset(i, j)];
    }

    MUDA_GENERIC T& operator()(int i, int j) noexcept
    {
        return m_data[cal_global_offset(i, j)];
    }

    MUDA_GENERIC int dim_i() const noexcept { return m_dim_i; }

    MUDA_GENERIC int dim_j(int i) const noexcept
    {
        check_dimi(i);
        return m_count[i];
    }

    MUDA_GENERIC int ndata(int i) const noexcept { return m_ndata; }

    // get the i-th row of the sparse 2d data structure
    MUDA_GENERIC dense1D<T> operator()(int i) noexcept
    {
        check_dimi(i);
        return dense1D<T>(m_data + m_begin[i], m_count[i]);
    }

  private:
    MUDA_GENERIC __forceinline__ void check_data() const noexcept
    {
        if constexpr(DEBUG_VIEWER)
        {
            muda_kernel_assert(m_data != nullptr, "cse[%s]: data is nullptr\n", name());
        }
    }

    MUDA_GENERIC __forceinline__ void check_dimi(int i) const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(i < 0 || i >= m_dim_i)
                muda_kernel_error("cse[%s]: out of range, i=(%d), dim_i=(%d)\n", name(), i, m_dim_i);
    }

    MUDA_GENERIC __forceinline__ void check_dimj(int i, int j, int dimj) const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(dimj < 0 || j < 0 || j >= dimj)
                muda_kernel_error(
                    "cse[%s]: out of range, ij=(%d,%d), dim=(%d,%d)\n", name(), i, j, m_dim_i, dimj);
    }

    MUDA_GENERIC __forceinline__ void check_global_offset(int i, int j, int dimj, int global_offset) const noexcept
    {
        if constexpr(DEBUG_VIEWER)
            if(global_offset < 0 || global_offset >= m_ndata)
                muda_kernel_error("cse[%s]: global_offset out of range, ij=(%d,%d), dim=(%d,%d), offset=(%d), ndata=(%d)\n",
                                  name(),
                                  i,
                                  j,
                                  m_dim_i,
                                  dimj,
                                  global_offset,
                                  m_ndata);
    }

    MUDA_GENERIC __forceinline__ int cal_global_offset(int i, int j) const noexcept
    {
        check_dimi(i);
        auto dimj = m_count[i];
        check_dimj(i, j, dimj);
        int global_offset = m_begin[i] + j;
        check_global_offset(i, j, dimj, global_offset);
        return global_offset;
    }
};
}  // namespace muda