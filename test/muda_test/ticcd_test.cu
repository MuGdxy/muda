#include <catch2/catch.hpp>
#include <type_traits>
#include <numeric>
#include <vector>
#include <algorithm>
#include <muda/muda.h>
#include "ccd_test_base.h"
#include <muda/pba/collision/ticcd.h>


using namespace muda;
namespace to = muda::thread_only;

using Scalar  = float;
using Vector3 = Eigen::Vector3<Scalar>;
using Array3  = Eigen::Array3<Scalar>;

enum class Type
{
    UnlimitedQueueSize,  // to use a property_queue with unlimited size
    LimitedQueueSize     // to use a property_queue with limited size
};

template <Type type = Type::UnlimitedQueueSize>
struct TiccdTestKernel
{
    MUDA_GENERIC TiccdTestKernel(bool              is_ee,
                                 dense2D<Vector3>  x,
                                 dense1D<uint32_t> results,
                                 dense1D<float>    tois)
        : x(x)
        , results(results)
        , tois(tois)
        , is_ee(is_ee)
    {
    }

    dense2D<Vector3>  x;
    dense1D<uint32_t> results;
    dense1D<float>    tois;
    bool              is_ee;

    MUDA_GENERIC void operator()(int i)
    {
        bool         res;
        Array3       err(-1, -1, -1);
        Scalar       ms = 1e-8;
        Scalar       toi;
        const Scalar tolerance = 1e-6;
        const Scalar t_max     = 1;
        Scalar       output_tolerance;
        const int    max_itr = 1e3;

        if(type == Type::UnlimitedQueueSize)
        {
            if(is_ee)
                res = ticcd<float>().edgeEdgeCCD(x(i, 0),
                                                 x(i, 1),
                                                 x(i, 2),
                                                 x(i, 3),
                                                 x(i, 4),
                                                 x(i, 5),
                                                 x(i, 6),
                                                 x(i, 7),
                                                 err,
                                                 ms,
                                                 toi,
                                                 tolerance,
                                                 t_max,
                                                 max_itr,
                                                 output_tolerance);
            else
                res = ticcd<float>().vertexFaceCCD(x(i, 0),
                                                   x(i, 1),
                                                   x(i, 2),
                                                   x(i, 3),
                                                   x(i, 4),
                                                   x(i, 5),
                                                   x(i, 6),
                                                   x(i, 7),
                                                   err,
                                                   ms,
                                                   toi,
                                                   tolerance,
                                                   t_max,
                                                   max_itr,
                                                   output_tolerance);
        }
        else if(type == Type::LimitedQueueSize)
        {
            const int maxQueueSize = 1024;

            const auto elementSize = sizeof(ticcd_alloc_elem_type<float>);

            using alloc = to::thread_stack_allocator<ticcd_alloc_elem_type<float>, maxQueueSize>;

            if(is_ee)
                res = ticcd<float, alloc>(maxQueueSize)
                          .edgeEdgeCCD(x(i, 0),
                                       x(i, 1),
                                       x(i, 2),
                                       x(i, 3),
                                       x(i, 4),
                                       x(i, 5),
                                       x(i, 6),
                                       x(i, 7),
                                       err,
                                       ms,
                                       toi,
                                       tolerance,
                                       t_max,
                                       max_itr,
                                       output_tolerance);
            else
                res = ticcd<float, alloc>(maxQueueSize)
                          .vertexFaceCCD(x(i, 0),
                                         x(i, 1),
                                         x(i, 2),
                                         x(i, 3),
                                         x(i, 4),
                                         x(i, 5),
                                         x(i, 6),
                                         x(i, 7),
                                         err,
                                         ms,
                                         toi,
                                         tolerance,
                                         t_max,
                                         max_itr,
                                         output_tolerance);
        }

        results(i) = res;
        tois(i)    = toi;
    }
};

template <Type type>
void ticcd_test(const std::string      file,
                bool                   is_ee,
                host_vector<uint32_t>& ground_thruth,
                host_vector<uint32_t>& h_results,
                host_vector<uint32_t>& d_results)
{
    host_vector<Eigen::Vector3f> X;
    read_ccd_csv(file, X, ground_thruth);

    auto resultSize = X.size() / 8;

    h_results.resize(resultSize);
    host_vector<float> h_tois(resultSize);

    auto ticcdKernel = TiccdTestKernel<type>(
        is_ee, make_dense2D(X, 8), make_viewer(h_results), make_viewer(h_tois));
    for(int i = 0; i < resultSize; i++)
    {
        ticcdKernel(i);
    }

    device_vector<Eigen::Vector3f> x = X;
    device_vector<uint32_t>        results(resultSize);
    device_vector<float>           tois(resultSize);

    parallel_for(32, 64)
        .apply(resultSize,
               TiccdTestKernel<type>(
                   is_ee, make_dense2D(x, 8), make_viewer(results), make_viewer(tois)))
        .wait();

    d_results = results;
}

const std::string ee = MUDA_TEST_DATA_DIR R"(/unit-tests/edge-edge/data_0_1.csv)";
const std::string vf = MUDA_TEST_DATA_DIR R"(/unit-tests/vertex-face/data_0_0.csv)";

TEST_CASE("ticcd", "[collide]")
{
    SECTION("limitedQueueSize-EE")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::LimitedQueueSize>(ee, true, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }

    SECTION("limitedQueueSize-VF")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::LimitedQueueSize>(vf, false, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }
}

TEST_CASE("ticcd-unlimited", "[.collide]")
{
    SECTION("UnlimitedQueueSize-EE")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::UnlimitedQueueSize>(ee, true, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }

    SECTION("UnlimitedQueueSize-VF")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::UnlimitedQueueSize>(vf, false, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }
}