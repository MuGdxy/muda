#include <catch2/catch.hpp>
#include <type_traits>
#include <numeric>
#include <vector>
#include <algorithm>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>
#include <muda/pba/collision/ticcd.h>
#include <iostream>
#include <fstream>

using namespace muda;
namespace to = muda::thread_only;

using Scalar  = float;
using Vector3 = Eigen::Vector3<Scalar>;
using Array3  = Eigen::Array3<Scalar>;

// read test data from csv file
inline void read_ticcd_csv(const std::string&            inputFileName,
                           host_vector<Eigen::Vector3f>& X,
                           host_vector<uint32_t>&        res)
{
    // be careful, there are n lines which means there are n/8 queries, but has
    // n results, which means results are duplicated
    std::vector<std::array<double, 3>> vs;
    vs.clear();
    std::ifstream infile;
    infile.open(inputFileName);
    std::array<double, 3> v;
    if(!infile.is_open())
    {
        throw std::exception("error path");
    }

    int l = 0;
    while(infile)  // there is input overload classfile
    {
        l++;
        std::string s;
        if(!getline(infile, s))
            break;
        if(s[0] != '#')
        {
            std::istringstream         ss(s);
            std::array<long double, 7> record;  // the first six are one vetex,
                                                // the seventh is the result
            int c = 0;
            while(ss)
            {
                std::string line;
                if(!getline(ss, line, ','))
                    break;
                try
                {
                    record[c] = std::stold(line);
                    c++;
                }
                catch(const std::invalid_argument e)
                {
                    std::cout << "NaN found in file " << inputFileName
                              << " line " << l << std::endl;
                    e.what();
                }
            }
            double x = record[0] / record[1];
            double y = record[2] / record[3];
            double z = record[4] / record[5];
            v[0]     = x;
            v[1]     = y;
            v[2]     = z;

            if(vs.size() % 8 == 0)
                res.push_back(record[6]);
            vs.push_back(v);
        }
    }
    X.resize(vs.size());
    for(int i = 0; i < vs.size(); i++)
    {
        X[i][0] = vs[i][0];
        X[i][1] = vs[i][1];
        X[i][2] = vs[i][2];
    }
    if(!infile.eof())
    {
        std::cerr << "Could not read file " << inputFileName << "\n";
    }
}

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
void ticcd_test(bool                   is_ee,
                host_vector<uint32_t>& ground_thruth,
                host_vector<uint32_t>& h_results,
                host_vector<uint32_t>& d_results)
{
    host_vector<Eigen::Vector3f> X;
    if(is_ee)
        read_ticcd_csv(MUDA_TEST_DATA_DIR R"(\unit-tests\edge-edge\data_0_1.csv)", X, ground_thruth);
    else
        read_ticcd_csv(MUDA_TEST_DATA_DIR R"(\unit-tests\vertex-face\data_0_0.csv)", X, ground_thruth);

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

    device_vector<uint32_t> lim_results(resultSize);
    d_results = results;
}

bool check_allow_false_positive(const host_vector<uint32_t>& ground_thruth,
                                const host_vector<uint32_t>& result)
{
    bool ret = true;
    for(size_t i = 0; i < ground_thruth.size(); ++i)
    {
        if(ground_thruth[i] > result[i])
        {
            ret = false;
            break;
        }
    }
    return ret;
}

TEST_CASE("ticcd", "[collide]")
{
    SECTION("limitedQueueSize-EE")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::LimitedQueueSize>(true, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }

    SECTION("limitedQueueSize-VF")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::LimitedQueueSize>(false, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }
}

TEST_CASE("ticcd-unlimited", "[.collide]") 
{
    SECTION("UnlimitedQueueSize-EE")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::UnlimitedQueueSize>(true, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }
    
    SECTION("UnlimitedQueueSize-VF")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::UnlimitedQueueSize>(false, ground_thruth, h_results, d_results);
        CHECK(check_allow_false_positive(ground_thruth, h_results));
        CHECK(check_allow_false_positive(ground_thruth, d_results));
    }
}