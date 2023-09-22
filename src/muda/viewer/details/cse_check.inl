#pragma once

namespace muda
{
namespace details
{
    MUDA_INLINE MUDA_GENERIC void cse_check_data(void* m_data, const char* m_name, const char* m_kernel_name) MUDA_NOEXCEPT
    {
        if constexpr(!::muda::NO_CHECK)
        {
            if(!(m_data != nullptr))
            {
                ::muda::print(
                    "(host):"
                    "%s(%d): %s <assert> "
                    "m_data != nullptr"
                    " failed."
                    "cse[%s:%s]: data is nullptr\n",
                    "D:\\MyStorage\\Project\\MiniMuda\\muda\\src\\muda\\viewer\\details\\cse_check.inl",
                    9,
                    __FUNCSIG__,
                    m_name,
                    m_kernel_name);
                if constexpr(::muda::TRAP_ON_ERROR)
                    ::muda::trap();
                ;
            }
        };
    }

    MUDA_INLINE MUDA_GENERIC void cse_check_dimi(int i, int m_dim_i, const char* m_name, const char* m_kernel_name) MUDA_NOEXCEPT
    {
        if(i < 0 || i >= m_dim_i)
            muda_kernel_error("cse[%s:%s]: out of range, i=(%d), dim_i=(%d)\n", m_name, m_kernel_name, i, m_dim_i);
    }

    MUDA_INLINE MUDA_GENERIC void cse_check_dimj(int i, int j, int dimj, int m_dim_i, const char* m_name, const char* m_kernel_name) MUDA_NOEXCEPT
    {

        if(dimj < 0 || j < 0 || j >= dimj)
            muda_kernel_error(
                "cse[%s:%s]: out of range, ij=(%d,%d), dim=(%d,%d)\n", m_name, m_kernel_name, i, j, m_dim_i, dimj);
    }

    MUDA_INLINE MUDA_GENERIC void cse_check_global_offset(
        int i, int j, int dimj, int global_offset, int m_dim_i, int m_ndata, const char* m_name, const char* m_kernel_name) MUDA_NOEXCEPT
    {

        if(global_offset < 0 || global_offset >= m_ndata)
            muda_kernel_error("cse[%s:%s]: global_offset out of range, ij=(%d,%d), dim=(%d,%d), offset=(%d), ndata=(%d)\n",
                              m_name, m_kernel_name,
                              i,
                              j,
                              m_dim_i,
                              dimj,
                              global_offset,
                              m_ndata);
    }
}  // namespace details
}  // namespace muda