#include <muda/ext/linear_system/matrix_format_converter_impl.h>

namespace muda
{
template <typename T, int N>
details::MatrixFormatConverter<T, N>& MatrixFormatConverter::impl()
{
    using namespace details;
    constexpr auto ask_data_type = cuda_data_type<T>();
    constexpr auto ask_N         = N;

    if(current)
    {
        if(current->data_type() == ask_data_type && current->dim() == ask_N)
        {
            return *static_cast<details::MatrixFormatConverter<T, N>*>(current);
        }
    }

    MatrixFormatConverterType type{ask_data_type, ask_N};
    auto                      it = m_impls.find(type);
    if(it != m_impls.end())
    {
        current = it->second.get();
        return *static_cast<details::MatrixFormatConverter<T, N>*>(current);
    }

    auto impl = std::make_unique<details::MatrixFormatConverter<T, N>>(m_handles);
    current = impl.get();
    m_impls.emplace(type, std::move(impl));
    return *static_cast<details::MatrixFormatConverter<T, N>*>(current);
}

inline MatrixFormatConverter::~MatrixFormatConverter() {}
}  // namespace muda

namespace muda
{
// Triplet -> BCOO
template <typename T, int N>
void MatrixFormatConverter::convert(const DeviceTripletMatrix<T, N>& from,
                                    DeviceBCOOMatrix<T, N>&          to)
{
    impl<T, N>().convert(from, to);
}

// BCOO -> Dense Matrix
template <typename T, int N>
void MatrixFormatConverter::convert(const DeviceBCOOMatrix<T, N>& from,
                                    DeviceDenseMatrix<T>&         to,
                                    bool clear_dense_matrix)
{
    impl<T, N>().convert(from, to, clear_dense_matrix);
}

// BCOO -> COO
template <typename T, int N>
void MatrixFormatConverter::convert(const DeviceBCOOMatrix<T, N>& from,
                                    DeviceCOOMatrix<T>&           to)
{
    impl<T, N>().convert(from, to);
}

// BCOO -> BSR
template <typename T, int N>
void MatrixFormatConverter::convert(const DeviceBCOOMatrix<T, N>& from,
                                    DeviceBSRMatrix<T, N>&        to)
{
    impl<T, N>().convert(from, to);
}

// Doublet -> BCOO
template <typename T, int N>
void MatrixFormatConverter::convert(const DeviceDoubletVector<T, N>& from,
                                    DeviceBCOOVector<T, N>&          to)
{
    impl<T, N>().convert(from, to);
}

// BCOO -> Dense Vector
template <typename T, int N>
void MatrixFormatConverter::convert(const DeviceBCOOVector<T, N>& from,
                                    DeviceDenseVector<T>&         to,
                                    bool clear_dense_vector)
{

    impl<T, N>().convert(from, to, clear_dense_vector);
}

// Doublet -> Dense Vector
template <typename T, int N>
void MatrixFormatConverter::convert(const DeviceDoubletVector<T, N>& from,
                                    DeviceDenseVector<T>&            to,
                                    bool clear_dense_vector)
{

    impl<T, N>().convert(from, to, clear_dense_vector);
}

// BSR -> CSR
template <typename T, int N>
void MatrixFormatConverter::convert(const DeviceBSRMatrix<T, N>& from,
                                    DeviceCSRMatrix<T>&          to)
{
    impl<T, N>().convert(from, to);
}

// Triplet -> COO
template <typename T>
void MatrixFormatConverter::convert(const DeviceTripletMatrix<T, 1>& from,
                                    DeviceCOOMatrix<T>&              to)
{
    impl<T, 1>().convert(from, to);
}

// COO -> Dense Matrix
template <typename T>
void MatrixFormatConverter::convert(const DeviceCOOMatrix<T>& from,
                                    DeviceDenseMatrix<T>&     to,
                                    bool clear_dense_matrix)
{
    impl<T, 1>().convert(from, to, clear_dense_matrix);
}

// COO -> CSR
template <typename T>
void MatrixFormatConverter::convert(const DeviceCOOMatrix<T>& from, DeviceCSRMatrix<T>& to)
{
    impl<T, 1>().convert(from, to);
}
template <typename T>
void MatrixFormatConverter::convert(DeviceCOOMatrix<T>&& from, DeviceCSRMatrix<T>& to)
{
    impl<T, 1>().convert(std::move(from), to);
}

// Doublet -> COO
template <typename T>
void MatrixFormatConverter::convert(const DeviceDoubletVector<T, 1>& from,
                                    DeviceCOOVector<T>&              to)
{
    impl<T, 1>().convert(from, to);
}

// COO -> Dense Vector
template <typename T>
void MatrixFormatConverter::convert(const DeviceCOOVector<T>& from,
                                    DeviceDenseVector<T>&     to,
                                    bool clear_dense_vector)
{
    impl<T, 1>().convert(from, to, clear_dense_vector);
}

// Doublet -> Dense Vector
template <typename T>
void MatrixFormatConverter::convert(const DeviceDoubletVector<T, 1>& from,
                                    DeviceDenseVector<T>&            to,
                                    bool clear_dense_vector)
{
    impl<T, 1>().convert(from, to, clear_dense_vector);
}
}  // namespace muda