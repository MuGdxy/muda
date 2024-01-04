#pragma once
namespace muda
{
namespace details
{
    template <typename T, int N>
    class MatrixFormatConverter
    {
      public:
    };
}  // namespace details

template <typename T, int N>
class MatrixFormatConverter
{
    details::MatrixFormatConverter<T, N> impl;
};
}  // namespace muda


#include "details/matrix_format_converter.inl"
