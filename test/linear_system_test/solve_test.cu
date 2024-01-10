#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <Eigen/Dense>
#include <muda/ext/linear_system.h>
using namespace muda;
using namespace Eigen;

//using T                = float;
//constexpr int BlockDim = 3;

template <typename T>
void test_linear_system_solve(int dim)
{
    Eigen::VectorX<T> ra = Eigen::VectorX<T>::Random(dim);
    for(int i = 0; i < dim; ++i)
    {
        if(ra(i) < 0.2)
            ra(i) = 0.0;  // make A sparse
    }


    // make A invertible
    Eigen::MatrixX<T> A_dense =
        ra * ra.transpose() + Eigen::MatrixX<T>::Identity(dim, dim);

    Eigen::VectorX<T> x = Eigen::VectorX<T>::Random(dim);

    // create random triplets
    std::vector<int> row_indices;
    row_indices.reserve(dim * 100);
    std::vector<int> col_indices;
    col_indices.reserve(dim * 100);
    std::vector<T> values;
    values.reserve(dim * 100);

    Eigen::VectorX<T> b = A_dense * x;

    // get non-zero entries
    for(int i = 0; i < dim; ++i)
    {
        for(int j = 0; j < dim; j++)
        {
            auto val = A_dense(i, j);
            if(val != 0.0)
            {
                row_indices.push_back(i);
                col_indices.push_back(j);
                values.push_back(A_dense(i, j));
            }
        }
    }


    {  // solve on host
        // solve dense
        Eigen::VectorX<T> x_eigen_dense = A_dense.colPivHouseholderQr().solve(b);
        REQUIRE(x.isApprox(x_eigen_dense));
    }

    // convert to gipc::DeviceDenseMatrix
    DeviceDenseMatrix<T> A_device = A_dense;
    // convert to gipc::DeviceDenseVector
    DeviceDenseVector<T> b_device = b;


    {  // solve on device
        LinearSystemContext ctx;
        // solve dense
        DeviceDenseVector<T> x_device(dim);

        x_device                 = b_device;
        DeviceDenseMatrix decomp = A_device;
        // solve dense
        ctx.solve(decomp.view(), x_device.view());
        ctx.sync();
        Eigen::VectorX<T> x_host;
        x_device.copy_to(x_host);
        REQUIRE(x.isApprox(x_host));

        // solve sparse
        x_device.fill(0);
        DeviceTripletMatrix<T, 1> A_triplet;
        A_triplet.reshape(dim, dim);
        A_triplet.resize_triplets(values.size());
        A_triplet.row_indices().copy_from(row_indices.data());
        A_triplet.col_indices().copy_from(col_indices.data());
        A_triplet.values().copy_from(values.data());

        DeviceCOOMatrix<T> A_coo;
        ctx.convert(A_triplet, A_coo);

        DeviceCSRMatrix<T> A_csr;
        ctx.convert(A_coo, A_csr);

        x_device.fill(0);
        ctx.solve(x_device.view(), A_csr.cview(), b_device.cview());
        ctx.sync();

        x_device.copy_to(x_host);
        REQUIRE(x.isApprox(x_host));
    }
}

TEST_CASE("solve", "[linear_system]")
{
    test_linear_system_solve<float>(10);
    test_linear_system_solve<float>(100);
    test_linear_system_solve<float>(1000);
}