#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/ext/linear_system.h>
using namespace muda;
using namespace Eigen;

//using T                = float;
//constexpr int BlockDim = 3;
template <typename T, int BlockDim>
void test_sparse_matrix(int block_row_size, int non_zero_block_count)
{
    int dimension = BlockDim * block_row_size;


    LinearSystemContext ctx;

    Eigen::MatrixX<T> dense_A = Eigen::MatrixX<T>::Zero(dimension, dimension);
    Eigen::VectorX<T> dense_x = Eigen::VectorX<T>::Zero(dimension);
    dense_x.setRandom();
    Eigen::MatrixX<T> host_A;

    // setup device vector
    DeviceDenseVector<T> x = dense_x;
    DeviceDenseVector<T> b(dimension);

    std::vector<int> row_indices(non_zero_block_count);
    std::vector<int> col_indices(non_zero_block_count);
    std::vector<Eigen::Matrix<T, BlockDim, BlockDim>> blocks(non_zero_block_count);

    for(int i = 0; i < non_zero_block_count; ++i)  // create random blocks
    {
        Eigen::Vector2f index =
            (Eigen::Vector2f::Random() + Eigen::Vector2f::Ones()) / 2.0f;
        row_indices[i] = index.x() * block_row_size;
        col_indices[i] = index.y() * block_row_size;
        if(row_indices[i] == block_row_size)
            row_indices[i] = block_row_size - 1;
        if(col_indices[i] == block_row_size)
            col_indices[i] = block_row_size - 1;
        blocks[i] = Eigen::Matrix<T, BlockDim, BlockDim>::Random();
    }

    for(int i = 0; i < non_zero_block_count; ++i)  // set dense matrix
    {
        Eigen::Block<Eigen::MatrixX<T>, -1, -1> block = dense_A.block(
            row_indices[i] * BlockDim, col_indices[i] * BlockDim, BlockDim, BlockDim);

        block += blocks[i];
    }

    Eigen::VectorX<T> ground_truth = dense_A * dense_x;
    Eigen::VectorX<T> host_b;

    DeviceTripletMatrix<T, BlockDim> A_triplet;
    A_triplet.reshape(block_row_size, block_row_size);
    A_triplet.resize_triplets(non_zero_block_count);

    A_triplet.block_row_indices().copy_from(row_indices.data());
    A_triplet.block_col_indices().copy_from(col_indices.data());
    A_triplet.block_values().copy_from(blocks.data());
    {
        ctx.spmv(A_triplet.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));
    }

    DeviceBCOOMatrix<T, BlockDim> A_bcoo;
    ctx.convert(A_triplet, A_bcoo);
    {
        b.fill(0);
        ctx.spmv(A_bcoo.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));
    }

    DeviceDenseMatrix<T> A;

    DeviceCOOMatrix<T> A_coo;
    ctx.convert(A_bcoo, A_coo);
    A.fill(0);
    ctx.convert(A_coo, A);
    {
        b.fill(0);
        ctx.spmv(A_coo.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));

        A.copy_to(host_A);
        REQUIRE(host_A.isApprox(dense_A));
    }


    ctx.convert(A_bcoo, A);
    {
        b.fill(0);
        ctx.mv(A.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));

        A.copy_to(host_A);
        REQUIRE(host_A.isApprox(dense_A));
    }

    DeviceBSRMatrix<T, BlockDim> A_bsr;
    ctx.convert(A_bcoo, A_bsr);
    {
        b.fill(0);
        ctx.spmv(A_bsr.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));
    }

    DeviceCSRMatrix<T> A_csr;
    ctx.convert(A_bsr, A_csr);
    {
        b.fill(0);
        ctx.spmv(A_csr.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));
    }

    A_csr.clear();
    ctx.convert(A_coo, A_csr);
    {
        b.fill(0);
        ctx.spmv(A_csr.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));
    }
}

TEST_CASE("spmv", "[linear_system]")
{
    test_sparse_matrix<float, 3>(10, 40);
    test_sparse_matrix<float, 3>(100, 400);
    test_sparse_matrix<float, 3>(1000, 4000);

    test_sparse_matrix<float, 12>(10, 24);
    test_sparse_matrix<float, 12>(100, 888);
    test_sparse_matrix<float, 12>(1000, 7992);
}