#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include "ccd_test_base.h"
#include <muda/pba/collision/accd.h>

using namespace muda;


using real = float;
using vec3 = Eigen::Vector3<real>;

struct ACCDKernel
{
    MUDA_GENERIC ACCDKernel(bool is_ee, dense2D<vec3> x, dense1D<uint32_t> results)
        : x(x)
        , results(results)
        , is_ee(is_ee)
    {
    }

    MUDA_GENERIC void operator()(int i)
    {
        real toi   = 1.0f;
        real eta   = 0.2;
        real thick = 0;

        if(is_ee)
        {
            vec3 a0s = x(i, 0);
            vec3 a1s = x(i, 1);
            vec3 b0s = x(i, 2);
            vec3 b1s = x(i, 3);

            vec3 a0e = x(i, 4);
            vec3 a1e = x(i, 5);
            vec3 b0e = x(i, 6);
            vec3 b1e = x(i, 7);

            results(i) = accd<real>().Edge_Edge_CCD(a0s,
                                                    a1s,
                                                    b0s,
                                                    b1s,
                                                    vec3(a0e - a0s),
                                                    vec3(a1e - a1s),
                                                    vec3(b0e - b0s),
                                                    vec3(b1e - b1s),
                                                    eta,
                                                    thick,
                                                    toi);
        }
        else
        {
            vec3 ps  = x(i, 0);
            vec3 t0s = x(i, 1);
            vec3 t1s = x(i, 2);
            vec3 t2s = x(i, 3);

            vec3 pe  = x(i, 4);
            vec3 t0e = x(i, 5);
            vec3 t1e = x(i, 6);
            vec3 t2e = x(i, 7);
            auto a     = accd<real>();
            results(i) = a.Point_Triangle_CCD(
                ps, t0s, t1s, t2s, 
                vec3(pe - ps), vec3(t0e - t0s), 
                vec3(t1e - t1s), vec3(t2e - t2s), 
                eta, thick, toi);
        }
    }

    dense2D<vec3>     x;
    dense1D<uint32_t> results;
    bool              is_ee;
};

void accd_test(const std::string      file,
               bool                   is_ee,
               host_vector<uint32_t>& ground_thruth,
               host_vector<uint32_t>& h_results,
               host_vector<uint32_t>& d_results)
{
    host_vector<vec3> X;
    read_ccd_csv(file, X, ground_thruth);
    auto resultSize = X.size() / 8;

    h_results.resize(resultSize);
    host_vector<float> h_tois(resultSize);

     HOST:
    auto accdKernel = ACCDKernel(is_ee, make_dense2D(X, 8), make_viewer(h_results));
    for(int i = 0; i < resultSize; i++)
        accdKernel(i);
	

    // DEVICE:
    device_vector<Eigen::Vector3f> x = X;
    device_vector<uint32_t>        results(resultSize);
    parallel_for(32, 64)
        .apply(resultSize, ACCDKernel(is_ee, make_dense2D(x, 8), make_viewer(results)))
        .wait();

    d_results = results;
}

const std::string ee = MUDA_TEST_DATA_DIR R"(/unit-tests/edge-edge/data_0_0.csv)";
const std::string vf = MUDA_TEST_DATA_DIR R"(/unit-tests/vertex-face/data_0_1.csv)";


TEST_CASE("accd", "[collision]")
{
    SECTION("Edge-Edge")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        accd_test(ee, true, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }

    SECTION("Vertex-Face")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        accd_test(vf, false, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }
}
