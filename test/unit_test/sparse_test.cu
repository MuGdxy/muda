//#include <catch2/catch.hpp>
//#include <muda/muda.h>
//#include <muda/container.h>
//#include <muda/buffer.h>
//#include <muda/blas/blas.h>
//
//using namespace muda;
//
//void spmv_test(size_t n, const HostVector<float>& in, HostVector<float>& out)
//{
//    Stream      s;
//    BlasContext ctx(s);
//
//    // allocate memory for our matrix
//
//    size_t rows   = 16;
//    size_t cols   = rows;
//    auto   rowPtr = DeviceVector<int>(rows + 1);
//    // diagonal sparse matrix
//    size_t nnz    = rows;
//    auto   colIdx = DeviceVector<int>(nnz);
//    auto   values = DeviceVector<float>(nnz);
//
//    auto x_buffer = DeviceVector<float>(cols);
//    x_buffer      = in;
//    auto x        = DenseVector<float>(x_buffer.data(), x_buffer.size());
//
//    auto y_buffer = DeviceVector<float>(cols);
//    auto y        = DenseVector<float>(y_buffer.data(), y_buffer.size());
//
//    //raii
//    auto M =
//        MatrixCSR<float>(rows, cols, nnz, rowPtr.data(), colIdx.data(), values.data());
//
//    DeviceVector<std::byte> buf;
//
//    on(s)
//        .next<ParallelFor>(32, 32)
//        .apply(nnz,
//               [=, M = make_viewer(M)] __device__(int i) mutable
//               {
//                   M.place_row(i, i);
//                   M.place_col(i, 0, i, 1.0f);  //create an identity matrix
//                   if(i == rows - 1)
//                       M.place_tail();  //place tail
//               })
//        .apply(nnz,
//               [M = make_viewer(M)] __device__(int i) mutable
//               {
//                   auto e = M.rw_elem(i, 0);
//                   e *= i;  //scale the diagonal with index i.
//                   assert(e == M(i, i));
//               })
//        .next<Blas>(ctx)
//        .spmv(M, x, y, buf)
//        .wait();
//    out = y_buffer;
//}
//
//
//TEST_CASE("spmv_test", "[sparse]")
//{
//    HostVector<float> in, out;
//    size_t            size = 16;
//    in.resize(size, 1);
//    spmv_test(size, in, out);
//    HostVector<float> ground_thruth = in;
//    for(size_t i = 0; i < ground_thruth.size(); ++i)
//    {
//        ground_thruth[i] = i;
//    }
//    REQUIRE(out == ground_thruth);
//}