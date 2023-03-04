#pragma once
#include <muda/tools/version.h>

#include <cublas.h>
#include "data_type_map.h"
#include <muda/check/checkCublas.h>
#include <muda/viewer/dense.h>
#include <muda/buffer/device_buffer.h>

namespace muda
{
template <typename T>
class dense_vec
{
  public:
    using value_type = T;
    dense_vec(value_type* data, size_t n)
        : m_data(data)
        , m_size(n)
    {
        checkCudaErrors(cusparseCreateDnVec(&m_dnvec, n, data, details::cudaDataTypeMap_v<T>));
    }

    ~dense_vec() 
    { 
        checkCudaErrors(cusparseDestroyDnVec(m_dnvec)); 
    }
    
    value_type*       data() { return m_data; }
    const value_type* data() const { return m_data; }
    size_t            size() const { return m_size; }
    operator cusparseDnVecDescr_t() { return m_dnvec; }

  private:

    cusparseDnVecDescr_t m_dnvec;
    value_type*          m_data;
    size_t               m_size;
};
}  // namespace muda


namespace muda
{
template <typename T>
inline __host__ auto make_dense(dense_vec<T>& v)
{
    return dense1D<T>(v.data(), v.size());
}

template <typename T>
inline __host__ auto make_viewer(dense_vec<T>& v)
{
    return make_dense(v);
}

template <typename T>
inline __host__ auto make_dense_vec(device_buffer<T>& buf)
{
    return dense_vec<T>(buf.data(), buf.size());
}
}  // namespace muda
