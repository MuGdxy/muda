#pragma once
#include <cublas.h>
#include <cusparse.h>

namespace muda
{
namespace details
{
    template <typename T>
    struct cudaDataTypeMap
    {
    };
    template <typename T>
    constexpr cudaDataType cudaDataTypeMap_v = cudaDataTypeMap<T>::value;

    template <>
    constexpr cudaDataType cudaDataTypeMap_v<float> = CUDA_R_32F;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<double> = CUDA_R_64F;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<int8_t> = CUDA_R_8I;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<uint8_t> = CUDA_R_8U;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<int16_t> = CUDA_R_16I;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<uint16_t> = CUDA_R_16U;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<int32_t> = CUDA_R_32I;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<uint32_t> = CUDA_R_32U;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<int64_t> = CUDA_R_64I;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<uint64_t> = CUDA_R_64U;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<half> = CUDA_R_16F;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<nv_bfloat16> = CUDA_R_16BF;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<cuComplex> = CUDA_C_32F;
    template <>
    constexpr cudaDataType cudaDataTypeMap_v<cuDoubleComplex> = CUDA_C_64F;

    // yet unknown
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_16BF;
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_16F;
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_8I;
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_8U;
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_16I;
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_16U;
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_32I;
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_32U;
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_64I;
    //template<> constexpr cudaDataType cudaDataTypeMap_v<	> = CUDA_C_64U;


    template <typename T>
    struct cusparseIndexTypeMap
    {
    };
    template <typename T>
    constexpr cusparseIndexType_t cusparseIndexTypeMap_v =
        cusparseIndexTypeMap<T>::value;

    template <>
    constexpr cusparseIndexType_t cusparseIndexTypeMap_v<uint16_t> = CUSPARSE_INDEX_16U;
    template <>
    constexpr cusparseIndexType_t cusparseIndexTypeMap_v<int32_t> = CUSPARSE_INDEX_32I;
    template <>
    constexpr cusparseIndexType_t cusparseIndexTypeMap_v<int64_t> = CUSPARSE_INDEX_64I;
}  // namespace details
}  // namespace muda