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
    y.buffer_view().fill(0);
    ParallelFor(0, stream())
        .kernel_name(__FUNCTION__)
        .apply(A.triplet_count(),
               [a = a,
                A = A.viewer().name("A"),
                x = x.viewer().name("x"),
                b = b,
                y = y.viewer().name("y")] __device__(int index) mutable
               {
                   auto&& [i, j, block]       = A(index);
                   auto                seg_x  = x.segment<N>(j * N);
                   auto                seg_y  = y.segment<N>(i * N);
                   Eigen::Vector<T, N> vec_x  = seg_x.as_eigen();
                   Eigen::Vector<T, N> vec_y  = seg_y.as_eigen();
                   auto                result = a * block * vec_x + b * vec_y;
                   seg_y.atomic_add(result.eval());
               });
}
template <typename T, int N>
void muda::LinearSystemContext::spmv(CTripletMatrixView<T, N>& A,
                                     CDenseVectorView<T>&      x,
                                     DenseVectorView<T>&       y)
{
    spmv<T, N>(T{1}, A, x, T{0}, y);
}
}  // namespace muda