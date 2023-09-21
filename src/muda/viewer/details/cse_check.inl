#pragma once

namespace muda
{
namespace details
{
    MUDA_INLINE MUDA_GENERIC void cse_check_data(void* m_data, const char* name) MUDA_NOEXCEPT
    {
        muda_kernel_assert(m_data != nullptr, "cse[%s]: data is nullptr\n", name);
    }

    MUDA_INLINE MUDA_GENERIC void cse_check_dimi(int i, int m_dim_i, const char* name) MUDA_NOEXCEPT
    {
        if(i < 0 || i >= m_dim_i)
            muda_kernel_error("cse[%s]: out of range, i=(%d), dim_i=(%d)\n", name, i, m_dim_i);
    }

    MUDA_INLINE MUDA_GENERIC void cse_check_dimj(int i, int j, int dimj, int m_dim_i, const char* name) MUDA_NOEXCEPT
    {

        if(dimj < 0 || j < 0 || j >= dimj)
            muda_kernel_error(
                "cse[%s]: out of range, ij=(%d,%d), dim=(%d,%d)\n", name, i, j, m_dim_i, dimj);
    }

    MUDA_INLINE MUDA_GENERIC void cse_check_global_offset(
        int i, int j, int dimj, int global_offset, int m_dim_i, int m_ndata, const char* name) MUDA_NOEXCEPT
    {

        if(global_offset < 0 || global_offset >= m_ndata)
            muda_kernel_error("cse[%s]: global_offset out of range, ij=(%d,%d), dim=(%d,%d), offset=(%d), ndata=(%d)\n",
                              name,
                              i,
                              j,
                              m_dim_i,
                              dimj,
                              global_offset,
                              m_ndata);
    }
}  // namespace details
}  // namespace muda