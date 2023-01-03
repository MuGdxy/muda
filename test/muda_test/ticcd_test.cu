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
    MUDA_GENERIC TiccdTestKernel(idxer1D<Vector3> x, idxer1D<uint32_t> results, idxer1D<float> tois)
        : x(x)
        , results(results)
        , tois(tois)
    {
    }

    idxer1D<Vector3>  x;
    idxer1D<uint32_t> results;
    idxer1D<float>    tois;

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
            res = to::ticcd<float>().edgeEdgeCCD(x(i * 8 + 0),
                                                 x(i * 8 + 1),
                                                 x(i * 8 + 2),
                                                 x(i * 8 + 3),
                                                 x(i * 8 + 4),
                                                 x(i * 8 + 5),
                                                 x(i * 8 + 6),
                                                 x(i * 8 + 7),
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
            const int maxQueueSize = 128;

            const auto elementSize = sizeof(to::ticcd_alloc_elem_type<float>);

            using alloc =
                to::thread_stack_allocator<to::ticcd_alloc_elem_type<float>, maxQueueSize>;

            res = to::ticcd<float, alloc>(maxQueueSize)
                      .edgeEdgeCCD(x(i * 8 + 0),
                                   x(i * 8 + 1),
                                   x(i * 8 + 2),
                                   x(i * 8 + 3),
                                   x(i * 8 + 4),
                                   x(i * 8 + 5),
                                   x(i * 8 + 6),
                                   x(i * 8 + 7),
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
void ticcd_test(host_vector<uint32_t>& ground_thruth,
                host_vector<uint32_t>& h_results,
                host_vector<uint32_t>& d_results)
{
    host_vector<Eigen::Vector3f> X;
    read_ticcd_csv(MUDA_TEST_DATA_DIR R"(\unit-tests\edge-edge\data_0_1.csv)", X, ground_thruth);


    auto resultSize = X.size() / 8;

    h_results.resize(resultSize);
    host_vector<float> h_tois(resultSize);

    host_for(host_type::host_sync)
        .apply(resultSize,
               TiccdTestKernel<type>(make_viewer(X), make_viewer(h_results), make_viewer(h_tois)))
        .wait();

    device_vector<Eigen::Vector3f> x = X;
    device_vector<uint32_t>        results(resultSize);
    device_vector<float>           tois(resultSize);

    parallel_for(32, 64)
        .apply(resultSize,
               TiccdTestKernel<type>(make_viewer(x), make_viewer(results), make_viewer(tois)))
        .wait();

    device_vector<uint32_t> lim_results(resultSize);
    d_results = results;
}

TEST_CASE("ticcd", "[collide]")
{
    SECTION("UnlimitedQueueSize")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::UnlimitedQueueSize>(ground_thruth, h_results, d_results);
        CHECK(ground_thruth == h_results);
        CHECK(ground_thruth == d_results);
    }

    SECTION("limitedQueueSize")
    {
        host_vector<uint32_t> ground_thruth, h_results, d_results;
        ticcd_test<Type::LimitedQueueSize>(ground_thruth, h_results, d_results);
        CHECK(ground_thruth == h_results);
        CHECK(ground_thruth == d_results);
    }
}
