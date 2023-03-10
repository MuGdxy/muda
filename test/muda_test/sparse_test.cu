#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>
#include <muda/blas/blas.h>

using namespace muda;

void spmv_test(size_t n, const host_vector<float>& in, host_vector<float>& out)
{
    stream      s;
    blasContext ctx(s);

    // allocate memory for our matrix

    size_t rows   = 16;
    size_t cols   = rows;
    auto   rowPtr = device_buffer<int>(s, rows + 1);
    // diagonal sparse matrix
    size_t nnz    = rows;
    auto   colIdx = device_buffer<int>(s, nnz);
    auto   values = device_buffer<float>(s, nnz);

    auto x_buffer = device_buffer<float>(s, cols);
    x_buffer.copy_from(in);
    auto x = dense_vec<float>(x_buffer.data(), x_buffer.size());

    auto y_buffer = device_buffer<float>(s, cols);
    auto y        = dense_vec<float>(y_buffer.data(), y_buffer.size());

    //raii
    auto M = matCSR<float>(
        rows, cols, nnz, rowPtr.data(), colIdx.data(), values.data());

    device_buffer<std::byte> buf(s);

    on(s)
        .next<parallel_for>(32, 32)
        .apply(nnz,
               [=, M = make_viewer(M)] __device__(int i) mutable
               {
                   M.place_row(i, i);
                   M.place_col(i, 0, i, 1.0f);  //create an identity matrix
                   if(i == rows - 1)
                       M.place_tail();  //place tail
               })
        .apply(nnz,
               [M = make_viewer(M)] __device__(int i) mutable
               {
                   auto e = M.rw_elem(i, 0);
                   e *= i;  //scale the diagonal with index i.
                   assert(e == M(i, i));
               })
        .next<blas>(ctx)
        .spmv(M, x, y, buf)
        .wait();

    y_buffer.copy_to(out);
}


TEST_CASE("spmv_test", "[sparse]")
{
    host_vector<float> in, out;
    size_t             size = 16;
    in.resize(size, 1);
    spmv_test(size, in, out);
    host_vector<float> ground_thruth = in;
    for(size_t i = 0; i < ground_thruth.size(); ++i)
    {
        ground_thruth[i] = i;
    }
    REQUIRE(out == ground_thruth);
}