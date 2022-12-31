#pragma once
#include "base.h"

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
class cse
{
    T* data_;

    int* begin_;
    int* count_;

    int ndata_;
    int dim_i_;

  public:
    MUDA_GENERIC cse() noexcept
        : data_(nullptr)
        , begin_(nullptr)
        , count_(nullptr)
        , ndata_(0)
        , dim_i_(0)
    {
    }

    MUDA_GENERIC cse(T* data, int ndata, int* begin, int* count, int dim_i) noexcept
        : data_(data)
        , begin_(begin)
        , count_(count)
        , ndata_(ndata)
        , dim_i_(dim_i)
    {
    }

    MUDA_GENERIC const T& operator()(int i, int j) const noexcept
    {
        return data_[cal_global_offset(i, j)];
    }

    MUDA_GENERIC T& operator()(int i, int j) noexcept
    {
        return data_[cal_global_offset(i, j)];
    }

    MUDA_GENERIC int dim_i() const noexcept { return dim_i_; }

    MUDA_GENERIC int dim_j(int i) const noexcept
    {
        check_dimi(i);
        return count_[i];
    }

    MUDA_GENERIC int ndata(int i) const noexcept { return ndata_; }

  private:
    MUDA_GENERIC __forceinline__ void check_data() const noexcept
    {
        if constexpr(debugViewers)
        {
            muda_kernel_assert(data_ != nullptr, "cse: data is nullptr\n");
        }
    }

    MUDA_GENERIC __forceinline__ void check_dimi(int i) const noexcept
    {
        if constexpr(debugViewers)
            if(i < 0 || i >= dim_i_)
                muda_kernel_error("cse: out of range, i=(%d), dim_i=(%d)\n", i, dim_i_);
    }

    MUDA_GENERIC __forceinline__ void check_dimj(int i, int j, int dimj) const noexcept
    {
        if constexpr(debugViewers)
            if(dimj < 0 || j < 0 || j >= dimj)
                muda_kernel_error("cse: out of range, ij=(%d,%d), dim=(%d,%d)\n", i, j, dim_i_, dimj);
    }

    MUDA_GENERIC __forceinline__ void check_global_offset(int i, int j, int dimj, int global_offset) const noexcept
    {
        if constexpr(debugViewers)
            if(global_offset < 0 || global_offset >= ndata_)
                muda_kernel_error("cse: global_offset out of range, ij=(%d,%d), dim=(%d,%d), offset=(%d), ndata=(%d)\n",
                                  i,
                                  j,
                                  dim_i_,
                                  dimj,
                                  global_offset,
                                  ndata_);
    }

    MUDA_GENERIC __forceinline__ int cal_global_offset(int i, int j) const noexcept
    {
        check_dimi(i);
        auto dimj = count_[i];
        check_dimj(i, j, dimj);
        int global_offset = begin_[i] + j;
        check_global_offset(i, j, dimj, global_offset);
        return global_offset;
    }
};
}  // namespace muda