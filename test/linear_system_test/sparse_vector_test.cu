#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/ext/linear_system.h>
using namespace muda;
using namespace Eigen;

//using T                = float;
//constexpr int BlockDim = 3;

template <typename T, int BlockDim>
void test_doublet_vector(int segment_size, int doublet_count)
{
    LinearSystemContext ctx;

    DeviceDoubletVector<T, BlockDim> doublet;
    doublet.reshape(segment_size);
    doublet.resize_doublet(doublet_count);

    std::vector<int>                        segment_indices(doublet_count);
    std::vector<Eigen::Vector<T, BlockDim>> segment_values(doublet_count);

    for(int i = 0; i < doublet_count; ++i)
    {
        Eigen::Vector2f index =
            (Eigen::Vector2f::Random() + Eigen::Vector2f::Ones()) / 2.0f;
        segment_indices[i] = index.x() * segment_size;
        if(segment_indices[i] == segment_size)
            segment_indices[i] = segment_size - 1;
        segment_values[i] = Eigen::Vector<T, BlockDim>::Ones();
    }

    doublet.segment_indices().copy_from(segment_indices.data());
    doublet.segment_values().copy_from(segment_values.data());

    Eigen::VectorX<T> ground_truth = Eigen::VectorX<T>::Zero(segment_size * BlockDim);

    for(int i = 0; i < doublet_count; ++i)
    {
        Eigen::Block<Eigen::VectorX<T>, -1, 1> seg =
            ground_truth.segment(segment_indices[i] * BlockDim, BlockDim);

        seg += segment_values[i];
    }

    Eigen::VectorX<T>             host_v;
    DeviceDenseVector<T>          v;
    DeviceBCOOVector<T, BlockDim> bcoo;
    ctx.convert(doublet, bcoo);

    v.fill(0);
    ctx.convert(doublet, v);
    v.copy_to(host_v);
    //std::cout << host_v.transpose() << std::endl;
    //std::cout << ground_truth.transpose() << std::endl;
    REQUIRE(host_v.isApprox(ground_truth));

    ctx.convert(bcoo, v);
    v.copy_to(host_v);
    //std::cout << host_v.transpose() << std::endl;
    //std::cout << ground_truth.transpose() << std::endl;
    REQUIRE(host_v.isApprox(ground_truth));
}

TEST_CASE("sparse_vector", "[linear_system]")
{
    test_doublet_vector<float, 3>(10, 5);
    test_doublet_vector<float, 3>(100, 400);
    test_doublet_vector<float, 3>(1000, 4000);

    test_doublet_vector<float, 12>(10, 24);
    test_doublet_vector<float, 12>(100, 888);
    test_doublet_vector<float, 12>(1000, 7992);
}