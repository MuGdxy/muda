#pragma once
#include <cublas.h>
#include "data_type_map.h"
#include "../check/checkCublas.h"
#include "../viewer/idxer.h"
#include "../buffer/device_buffer.h"

namespace muda
{
namespace dense
{
    template <typename T>
    class vec
    {
      public:
        using value_type = T;
        vec(value_type* data, size_t n)
            : data_(data)
            , size_(n)
        {
            checkCudaErrors(
                cusparseCreateDnVec(&dnVec_, n, data, details::cudaDataTypeMap_v<T>));
        }
        ~vec() { checkCudaErrors(cusparseDestroyDnVec(dnVec_)); }
        value_type*       data() { return data_; }
        const value_type* data() const { return data_; }
        size_t            size() const { return size_; }
                          operator cusparseDnVecDescr_t() { return dnVec_; }

      private:
        cusparseDnVecDescr_t dnVec_;
        value_type*          data_;
        size_t               size_;
    };

}  // namespace dense
}  // namespace muda


namespace muda
{
template <typename T>
inline __host__ auto make_idxer(dense::vec<T>& v)
{
    return idxer1D<T>(v.data(), v.size());
}

template <typename T>
inline __host__ auto make_viewer(dense::vec<T>& v)
{
    return make_idxer(v);
}

template <typename T>
inline __host__ auto make_vec(device_buffer<T>& buf)
{
    return dense::vec<T>(buf.data(), buf.size());
}
}  // namespace muda