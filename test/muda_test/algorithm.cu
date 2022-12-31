#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <numeric>
#include <algorithm>
using namespace muda;

#include <muda/algorithm/device_scan.h>
//prefix sum
void prefix_sum(const host_vector<int>& input_data,
                host_vector<int>&       exclusive,
                host_vector<int>&       inclusive)
{
    size_t             count = input_data.size();
    device_vector<int> input = input_data;
    device_vector<int> excl(count, 0);
    device_vector<int> incl(count, 0);

    device_vector<int> keyout(count);
    device_vector<int> valueout(count);

    device_buffer buf;
    DeviceScan()
        .ExclusiveSum(buf, data(excl), data(input), count)
        .InclusiveSum(buf, data(incl), data(input), count)
        .wait();

    exclusive = excl;
    inclusive = incl;
}

TEST_CASE("prefix_sum", "[algorithm]")
{
    int              count = 99;
    host_vector<int> input(count, 1);
    host_vector<int> gt_ex(count, 0);
    host_vector<int> gt_in(count, 0);
    thrust::exclusive_scan(input.begin(), input.end(), gt_ex.begin());
    thrust::inclusive_scan(input.begin(), input.end(), gt_in.begin());
    host_vector<int> ex(count, 1);
    host_vector<int> in(count, 1);
    prefix_sum(input, ex, in);
    REQUIRE(ex == gt_ex);
    REQUIRE(in == gt_in);
}

#include <muda/algorithm/device_radix_sort.h>
#include <thrust/sort.h>
//radix sort
void radix_sort(const host_vector<int>& key_in,
                host_vector<int>&       key_out,
                host_vector<int>&       gt_key_out,
                const host_vector<int>& value_in,
                host_vector<int>&       value_out,
                host_vector<int>&       gt_value_out)
{
    size_t count = key_in.size();
    //copy key_in/value_in to d_key_in,d_value_in
    device_vector<int> d_key_in   = key_in;
    device_vector<int> d_value_in = value_in;

    //create d_keyout and d_value_out
    device_vector<int> d_key_out(count);
    device_vector<int> d_value_out(count);
    device_buffer      buf;
    DeviceRadixSort()
        .SortPairs(buf, data(d_key_out), data(d_value_out), data(d_key_in), data(d_value_in), count)
        .wait();
    //copy back to host
    key_out   = d_key_out;
    value_out = d_value_out;

    gt_key_out   = key_in;
    gt_value_out = value_in;
    thrust::sort_by_key(gt_key_out.begin(), gt_key_out.end(), gt_value_out.begin());
}

TEST_CASE("radix_sort", "[algorithm]")
{
    int              count = 32;
    host_vector<int> key_in(count);
    host_vector<int> value_in(count);

    //generate random data in key_in
    std::generate(key_in.begin(), key_in.end(), std::rand);
    //generate random data in value_in
    std::generate(value_in.begin(), value_in.end(), std::rand);

    host_vector<int> key_out;
    host_vector<int> gt_key_out;
    host_vector<int> value_out;
    host_vector<int> gt_value_out;

    radix_sort(key_in, key_out, gt_key_out, value_in, value_out, gt_value_out);
    REQUIRE(key_out == gt_key_out);
    REQUIRE(value_out == gt_value_out);
}

#include <muda/algorithm/device_reduce.h>
#undef max
void reduce(host_vector<int> in, int& out, int& gt_out)
{
    device_buffer   buf;
    auto            d_in = to_device(in);
    device_var<int> d_out;
    DeviceReduce()
        .Reduce(
            buf,
            data(d_out),
            data(d_in),
            d_in.size(),
            [] __device__(int a, int b) { return a > b ? a : b; },
            INT_MIN)
        .wait();
    out = gt_out;

    gt_out = *std::max(in.begin(), in.end());
}

TEST_CASE("reduce", "[algorithm]")
{
    int count = 100;
    //generate random vector
    host_vector<int> in(count);
    std::generate(in.begin(), in.end(), std::rand);
    int out;
    int gt_out;
    reduce(in, out, gt_out);
}