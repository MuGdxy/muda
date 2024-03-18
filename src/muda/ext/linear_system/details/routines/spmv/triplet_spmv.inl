#include <muda/ext/eigen.h>
namespace muda
{
//using T         = float;
//constexpr int N = 3;

template <typename T, int N>
void LinearSystemContext::spmv(const T&                 a,
                               CTripletMatrixView<T, N> A,
                               CDenseVectorView<T>      x,
                               const T&                 b,
                               DenseVectorView<T>&      y)
{
    using namespace muda;

    MUDA_ASSERT(A.extent() == A.total_extent() && A.triplet_count() == A.total_triplet_count(),
                "submatrix or subview of a Triplet Matrix is not allowed in SPMV!");

    MUDA_ASSERT(A.total_block_cols() * N == x.size() && A.total_block_rows() * N == y.size(),
                "Dimension mismatch in SPMV!");

    if(b != T{0})
	{
        ParallelFor(0, stream())
            .kernel_name(__FUNCTION__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
	}
    else
    {
        BufferLaunch(stream()).fill(y.buffer_view(), T{0});
    }

    ParallelFor(0, stream())
        .kernel_name(__FUNCTION__)
        .apply(A.triplet_count(),
               [a = a,
                A = A.viewer().name("A"),
                x = x.viewer().name("x"),
                b = b,
                y = y.viewer().name("y")] __device__(int index) mutable
               {
                   auto&& [i, j, block] = A(index);
                   auto seg_x           = x.segment<N>(j * N);

                   Eigen::Vector<T, N> vec_x  = seg_x.as_eigen();
                   auto                result = a * block * vec_x;

                   auto seg_y = y.segment<N>(i * N);
                   seg_y.atomic_add(result.eval());
               });

    //if(b != T{0})
    //{
    //    ParallelFor(0, stream())
    //        .kernel_name(__FUNCTION__)
    //        .apply(y.size(),
    //               [b = b, t = t.viewer().name("t"), y = y.viewer().name("y")] __device__(
    //                   int i) mutable { y(i) += b * t(i); });
    //}
}
template <typename T, int N>
void muda::LinearSystemContext::spmv(CTripletMatrixView<T, N> A,
                                     CDenseVectorView<T>      x,
                                     DenseVectorView<T>       y)
{
    spmv<T, N>(T{1}, A, x, T{0}, y);
}
}  // namespace muda