#include <numeric>
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/cub/cub.h>
#undef max
#undef min
using namespace muda;

struct Sortable
{
    int id = -1;
    int data;
};

__host__ __device__ int operator<(const Sortable& lhs, const Sortable& rhs)
{
    return lhs.id < rhs.id;
}

__host__ __device__ int operator>(const Sortable& lhs, const Sortable& rhs)
{
    return lhs.id > rhs.id;
}

 

struct Reducable
{
    int id = -1;
    int data;
};

__host__ __device__ int operator==(const Reducable& lhs, const Reducable& rhs)
{
    return lhs.id == rhs.id;
}

void device_reduce_reduce(Reducable& h_output, Reducable& gt_output)
{

    size_t size = 100;

    std::vector<Reducable> gt_input(size);

    DeviceBuffer<Reducable> input;
    DeviceVar<Reducable>    output;

    std::for_each(gt_input.begin(),
                  gt_input.end(),
                  [](Reducable& r) { r.id = std::rand() % 101; });
    input = gt_input;

    on(nullptr)
        .next<DeviceReduce>()
        .Reduce(

            input.data(),
            output.data(),
            size,
            [] __host__ __device__(const Reducable& l, const Reducable& r) -> Reducable
            { return l.id > r.id ? l : r; },
            Reducable{})
        .wait();

    gt_output = *std::max_element(gt_input.begin(),
                                  gt_input.end(),
                                  [](const Reducable& l, const Reducable& r)
                                  { return l.id < r.id; });

    h_output = output;
}


void device_reduce_min(int& h_output, int& gt_output)
{

    size_t size = 100;

    std::vector<int> gt_input(size);

    DeviceBuffer<int> input;
    DeviceVar<int>    output;

    std::for_each(
        gt_input.begin(), gt_input.end(), [](int& r) { r = std::rand() % 101; });
    input = gt_input;

    on().next<DeviceReduce>().Min(input.data(), output.data(), size).wait();

    gt_output = *std::min_element(gt_input.begin(), gt_input.end());

    h_output = output;
}

void device_reduce_max(int& h_output, int& gt_output)
{

    size_t size = 100;

    std::vector<int> gt_input(size);

    DeviceBuffer<int> input;
    DeviceVar<int>    output;

    std::for_each(
        gt_input.begin(), gt_input.end(), [](int& r) { r = std::rand() % 101; });
    input = gt_input;

    on().next<DeviceReduce>().Max(input.data(), output.data(), size).wait();

    gt_output = *std::max_element(gt_input.begin(), gt_input.end());

    h_output = output;
}


void device_reduce_sum(int& h_output, int& gt_output)
{

    size_t size = 100;

    std::vector<int> gt_input(size);

    DeviceBuffer<int> input;
    DeviceVar<int>    output;

    std::for_each(
        gt_input.begin(), gt_input.end(), [](int& r) { r = std::rand() % 101; });
    input = gt_input;

    on().next<DeviceReduce>().Sum(input.data(), output.data(), size).wait();

    gt_output = std::accumulate(gt_input.begin(), gt_input.end(), 0.0f);

    h_output = output;
}

void device_reduce_argmin(int& h_output, int& gt_output)
{
    using KVP = cub::KeyValuePair<int, int>;

    size_t size = 100;

    std::vector<int>  gt_input(size);
    DeviceBuffer<int> input;
    DeviceVar<KVP>      output;
    KVP                 h_output_kvp;


    std::for_each(
        gt_input.begin(), gt_input.end(), [](int& r) { r = std::rand() % 101; });
    input = gt_input;

    // using std to get the index of the min element
    gt_output = std::min_element(gt_input.begin(), gt_input.end()) - gt_input.begin();


    on().next<DeviceReduce>().ArgMin(input.data(), output.data(), size).wait();

    h_output_kvp = output;
    h_output     = h_output_kvp.key;
}

void device_reduce_argmax(int& h_output, int& gt_output)
{
    using KVP = cub::KeyValuePair<int, int>;

    size_t size = 100;

    std::vector<int>  gt_input(size);
    DeviceBuffer<int> input;
    DeviceVar<KVP>      output;
    KVP                 h_output_kvp;

    std::for_each(
        gt_input.begin(), gt_input.end(), [](int& r) { r = std::rand() % 101; });
    input = gt_input;

    // using std to get the index of the max element
    gt_output = std::max_element(gt_input.begin(), gt_input.end()) - gt_input.begin();


    on().next<DeviceReduce>().ArgMax(input.data(), output.data(), size).wait();

    h_output_kvp = output;
    h_output     = h_output_kvp.key;
}

// CustomMin functor
struct CustomMin
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
    {
        return (b < a) ? b : a;
    }
};

void device_reduce_reduce_by_key(std::vector<int>& h_unique_out,
                                 std::vector<int>& h_aggregates_out,
                                 int&              h_num_runs_out,
                                 std::vector<int>& gt_unique_out,
                                 std::vector<int>& gt_aggregates_out,
                                 int&              gt_num_runs_out)
{

    size_t size = 100;

    std::vector<int>  gt_input(size);
    DeviceBuffer<int> input;

    // Declare, allocate, and initialize device-accessible pointers for input and output
    int num_items = 8;  // e.g., 8

    std::vector<int>  keys_in          = {0, 2, 2, 9, 5, 5, 5, 8};
    std::vector<int>  values_in        = {0, 7, 1, 6, 2, 5, 3, 4};
    DeviceBuffer<int> d_keys_in        = keys_in;
    DeviceBuffer<int> d_values_in      = values_in;
    DeviceBuffer<int> d_unique_out     = keys_in;
    DeviceBuffer<int> d_aggregates_out = keys_in;
    DeviceVar<int>    d_num_runs_out;
    CustomMin         reduction_op;

    on().next<DeviceReduce>()
        .ReduceByKey(d_keys_in.data(),
                     d_unique_out.data(),
                     d_values_in.data(),
                     d_aggregates_out.data(),
                     d_num_runs_out.data(),
                     reduction_op,
                     num_items)
        .wait();

    d_unique_out.resize(d_num_runs_out);
    d_aggregates_out.resize(d_num_runs_out);

    d_unique_out.copy_to(h_unique_out);
    d_aggregates_out.copy_to(h_aggregates_out);
    h_num_runs_out = d_num_runs_out;

    gt_unique_out     = std::vector<int>{0, 2, 9, 5, 8};
    gt_aggregates_out = std::vector<int>{0, 1, 6, 2, 4};
    gt_num_runs_out   = 5;
}

TEST_CASE("device_reduce", "[cub]")
{
    SECTION("Reduce")
    {
        Reducable h_output;
        Reducable gt_output;
        device_reduce_reduce(h_output, gt_output);
        REQUIRE(h_output.id == gt_output.id);
    }

    SECTION("Min")
    {
        int h_output;
        int gt_output;
        device_reduce_min(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("Max")
    {
        int h_output;
        int gt_output;
        device_reduce_max(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("Sum")
    {
        int h_output;
        int gt_output;
        device_reduce_sum(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("ArgMin")
    {
        int h_output;
        int gt_output;
        device_reduce_argmin(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("ArgMax")
    {
        int h_output;
        int gt_output;
        device_reduce_argmax(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("ReduceByKey")
    {
        std::vector<int> h_unique_out;
        std::vector<int> h_aggregates_out;
        int              h_num_runs_out;
        std::vector<int> gt_unique_out;
        std::vector<int> gt_aggregates_out;
        int              gt_num_runs_out;
        device_reduce_reduce_by_key(
            h_unique_out, h_aggregates_out, h_num_runs_out, gt_unique_out, gt_aggregates_out, gt_num_runs_out);
        REQUIRE(h_unique_out == gt_unique_out);
        REQUIRE(h_aggregates_out == gt_aggregates_out);
        REQUIRE(h_num_runs_out == gt_num_runs_out);
    }
}

void device_scan_inclusive_sum(std::vector<int>& h_output, std::vector<int>& gt_output)
{

    size_t size = 100;

    std::vector<int> gt_input(size);
    gt_output.resize(size);
    DeviceBuffer<int> input(size);
    DeviceBuffer<int> output(size);

    std::for_each(
        gt_input.begin(), gt_input.end(), [](int& r) { r = std::rand() % 101; });
    input = gt_input;

    // using std to get the inclusive sum
    std::partial_sum(gt_input.begin(), gt_input.end(), gt_output.begin());

    on().next<DeviceScan>().InclusiveSum(input.data(), output.data(), size).wait();

    output.copy_to(h_output);
}

void device_scan_exclusive_sum(std::vector<int>& h_output, std::vector<int>& gt_output)
{

    size_t size = 100;

    std::vector<int> gt_input(size);
    gt_output.resize(size);
    DeviceBuffer<int> input(size);
    DeviceBuffer<int> output(size);

    std::for_each(
        gt_input.begin(), gt_input.end(), [](int& r) { r = std::rand() % 101; });
    input = gt_input;

    // using std to get the exclusive sum
    gt_output[0] = 0;
    std::partial_sum(gt_input.begin(), gt_input.end() - 1, gt_output.begin() + 1);

    on().next<DeviceScan>().ExclusiveSum(input.data(), output.data(), size).wait();

    output.copy_to(h_output);
}

void device_scan_inclusive_scan(std::vector<int>& h_output, std::vector<int>& gt_output)
{

    size_t size = 100;

    std::vector<int> gt_input(size);
    gt_output.resize(size);
    DeviceBuffer<int> input(size);
    DeviceBuffer<int> output(size);

    std::for_each(
        gt_input.begin(), gt_input.end(), [](int& r) { r = std::rand() % 101; });
    input = gt_input;

    // using std to get the inclusive scan
    std::partial_sum(gt_input.begin(), gt_input.end(), gt_output.begin());

    on().next<DeviceScan>()
        .InclusiveScan(

            input.data(),
            output.data(),
            [] __host__ __device__(const int& a, const int& b)
            { return a + b; },
            size)
        .wait();

    output.copy_to(h_output);
}

void device_scan_exclusive_scan(std::vector<int>& h_output, std::vector<int>& gt_output)
{

    size_t size = 100;

    std::vector<int> gt_input(size);
    gt_output.resize(size);
    DeviceBuffer<int> input(size);
    DeviceBuffer<int> output(size);

    std::for_each(
        gt_input.begin(), gt_input.end(), [](int& r) { r = std::rand() % 101; });
    input = gt_input;

    // using std to get the exclusive scan
    gt_output[0] = 0;
    std::partial_sum(gt_input.begin(), gt_input.end() - 1, gt_output.begin() + 1);

    on().next<DeviceScan>()
        .ExclusiveScan(

            input.data(),
            output.data(),
            [] __host__ __device__(const int& a, const int& b)
            { return a + b; },
            0.0f,
            size)
        .wait();

    output.copy_to(h_output);
}


void device_scan_exclusive_sum_by_key(std::vector<int>& h_values_out,
                                      std::vector<int>& gt_values_out)
{

    size_t size = 8;

    std::vector<int> h_keys_in   = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    std::vector<int> h_values_in = std::vector{0, 7, 1, 6, 2, 5, 3, 4};
    gt_values_out                = std::vector{0, 0, 7, 0, 0, 2, 7, 0};

    DeviceBuffer<int> d_keys_in   = h_keys_in;
    DeviceBuffer<int> d_values_in = h_values_in;
    DeviceBuffer<int> d_keys_out(size);
    DeviceBuffer<int> d_values_out(size);

    on().next<DeviceScan>()
        .ExclusiveSumByKey(d_keys_in.data(), d_values_in.data(), d_values_out.data(), size)
        .wait();

    d_values_out.copy_to(h_values_out);
}


void device_scan_inclusive_sum_by_key(std::vector<int>& h_values_out,
                                      std::vector<int>& gt_values_out)
{

    size_t size = 8;

    std::vector<int> h_keys_in   = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    std::vector<int> h_values_in = std::vector{0, 7, 1, 6, 2, 5, 3, 4};
    gt_values_out                = std::vector{0, 7, 8, 6, 2, 7, 10, 4};

    DeviceBuffer<int> d_keys_in   = h_keys_in;
    DeviceBuffer<int> d_values_in = h_values_in;
    DeviceBuffer<int> d_keys_out(size);
    DeviceBuffer<int> d_values_out(size);

    on().next<DeviceScan>()
        .InclusiveSumByKey(d_keys_in.data(), d_values_in.data(), d_values_out.data(), size)
        .wait();

    d_values_out.copy_to(h_values_out);
}


void device_scan_exclusive_scan_by_key(std::vector<int>& h_values_out,
                                       std::vector<int>& gt_values_out)
{

    size_t size = 8;

    std::vector<int> h_keys_in   = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    std::vector<int> h_values_in = std::vector{0, 7, 1, 6, 2, 5, 3, 4};
    gt_values_out                = std::vector{0, 0, 7, 0, 0, 2, 7, 0};

    DeviceBuffer<int> d_keys_in   = h_keys_in;
    DeviceBuffer<int> d_values_in = h_values_in;
    DeviceBuffer<int> d_keys_out(size);
    DeviceBuffer<int> d_values_out(size);

    on().next<DeviceScan>()
        .ExclusiveScanByKey(

            d_keys_in.data(),
            d_values_in.data(),
            d_values_out.data(),
            [] __host__ __device__(const int& a, const int& b) -> int
            { return a + b; },
            0,
            size)
        .wait();

    d_values_out.copy_to(h_values_out);
}


void device_scan_inclusive_scan_by_key(std::vector<int>& h_values_out,
                                       std::vector<int>& gt_values_out)
{

    size_t size = 8;

    std::vector<int> h_keys_in   = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    std::vector<int> h_values_in = std::vector{0, 7, 1, 6, 2, 5, 3, 4};
    gt_values_out                = std::vector{0, 7, 8, 6, 2, 7, 10, 4};

    DeviceBuffer<int> d_keys_in   = h_keys_in;
    DeviceBuffer<int> d_values_in = h_values_in;
    DeviceBuffer<int> d_keys_out(size);
    DeviceBuffer<int> d_values_out(size);


    on().next<DeviceScan>()
        .InclusiveScanByKey(

            d_keys_in.data(),
            d_values_in.data(),
            d_values_out.data(),
            [] __host__ __device__(const int& a, const int& b) { return a + b; },
            size)
        .wait();

    d_values_out.copy_to(h_values_out);
}

TEST_CASE("device_scan", "[cub]")
{
    SECTION("InclusiveSum")
    {
        std::vector<int> h_output, gt_output;
        device_scan_inclusive_sum(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("ExclusiveSum")
    {
        std::vector<int> h_output, gt_output;
        device_scan_exclusive_sum(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }


    SECTION("InclusiveScan")
    {
        std::vector<int> h_output, gt_output;
        device_scan_inclusive_scan(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }


    SECTION("ExclusiveScan")
    {
        std::vector<int> h_output, gt_output;
        device_scan_exclusive_scan(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("ExclusiveSumByKey")
    {
        std::vector<int> h_values_out, gt_values_out;
        device_scan_exclusive_sum_by_key(h_values_out, gt_values_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("InclusiveSumByKey")
    {
        std::vector<int> h_values_out, gt_values_out;
        device_scan_inclusive_sum_by_key(h_values_out, gt_values_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("ExclusiveScanByKey")
    {
        std::vector<int> h_values_out, gt_values_out;
        device_scan_exclusive_scan_by_key(h_values_out, gt_values_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("InclusiveScanByKey")
    {
        std::vector<int> h_values_out, gt_values_out;
        device_scan_inclusive_scan_by_key(h_values_out, gt_values_out);
        REQUIRE(h_values_out == gt_values_out);
    }
}


void device_run_length_encode_encode(std::vector<int>& h_unique_out,
                                     std::vector<int>& h_counts_out,
                                     int&              h_num_runs_out,
                                     std::vector<int>& gt_unique_out,
                                     std::vector<int>& gt_counts_out,
                                     int&              gt_num_runs_out)
{

    size_t size = 8;

    std::vector<int> h_input = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    gt_unique_out            = std::vector{0, 2, 9, 5, 8};
    gt_counts_out            = std::vector{1, 2, 1, 3, 1};
    gt_num_runs_out          = 5;

    DeviceBuffer<int> d_input = h_input;
    DeviceBuffer<int> d_unique_out(size);
    DeviceBuffer<int> d_counts_out(size);
    DeviceVar<int>    d_num_runs_out;

    on().next<DeviceRunLengthEncode>()
        .Encode(d_input.data(),
                d_unique_out.data(),
                d_counts_out.data(),
                d_num_runs_out.data(),
                size)
        .wait();

    d_unique_out.resize(d_num_runs_out);
    d_counts_out.resize(d_num_runs_out);

    d_unique_out.copy_to(h_unique_out);
    d_counts_out.copy_to(h_counts_out);

    h_num_runs_out = d_num_runs_out;
}

void device_run_length_encode_non_trivial_runs(std::vector<int>& h_offsets_out,
                                               std::vector<int>& h_counts_out,
                                               int&              h_num_runs_out,
                                               std::vector<int>& gt_offsets_out,
                                               std::vector<int>& gt_counts_out,
                                               int& gt_num_runs_out)
{

    size_t size = 8;

    std::vector<int> h_input = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    gt_offsets_out           = std::vector{1, 4};
    gt_counts_out            = std::vector{2, 3};
    gt_num_runs_out          = 2;

    DeviceBuffer<int> d_input = h_input;
    DeviceBuffer<int> d_offsets_out(size);
    DeviceBuffer<int> d_counts_out(size);
    DeviceVar<int>    d_num_runs_out;

    on().next<DeviceRunLengthEncode>()
        .NonTrivialRuns(d_input.data(),
                        d_offsets_out.data(),
                        d_counts_out.data(),
                        d_num_runs_out.data(),
                        size)
        .wait();

    d_offsets_out.resize(d_num_runs_out);
    d_counts_out.resize(d_num_runs_out);

    d_offsets_out.copy_to(h_offsets_out);
    d_counts_out.copy_to(h_counts_out);
    h_num_runs_out = d_num_runs_out;
}

TEST_CASE("device_run_length_encode", "[cub]")
{
    SECTION("Encode")
    {
        std::vector<int> h_unique_out, h_counts_out, gt_unique_out, gt_counts_out;
        int h_num_runs_out, gt_num_runs_out;
        device_run_length_encode_encode(
            h_unique_out, h_counts_out, h_num_runs_out, gt_unique_out, gt_counts_out, gt_num_runs_out);
        REQUIRE(h_unique_out == gt_unique_out);
        REQUIRE(h_counts_out == gt_counts_out);
        REQUIRE(h_num_runs_out == gt_num_runs_out);
    }

    SECTION("NonTrivialRuns")
    {
        std::vector<int> h_offsets_out, h_counts_out, gt_offsets_out, gt_counts_out;
        int h_num_runs_out, gt_num_runs_out;
        device_run_length_encode_non_trivial_runs(
            h_offsets_out, h_counts_out, h_num_runs_out, gt_offsets_out, gt_counts_out, gt_num_runs_out);
        REQUIRE(h_offsets_out == gt_offsets_out);
        REQUIRE(h_counts_out == gt_counts_out);
        REQUIRE(h_num_runs_out == gt_num_runs_out);
    }
}

void device_radix_sort_sort_pairs(std::vector<int>&   h_keys_out,
                                  std::vector<int>& h_values_out,
                                  std::vector<int>&   gt_keys_out,
                                  std::vector<int>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int>   h_keys_in(size);
    std::vector<int> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort
    gt_keys_out   = h_keys_in;
    gt_values_out = h_values_in;

    std::vector<Sortable> sortable(size);

    for(size_t i = 0; i < size; ++i)
    {
        sortable[i].id   = h_keys_in[i];
        sortable[i].data = h_values_in[i];
    }

    std::stable_sort(sortable.begin(), sortable.end());
    for(size_t i = 0; i < size; ++i)
    {
        gt_keys_out[i]   = sortable[i].id;
        gt_values_out[i] = sortable[i].data;
    }

    // Sort input data using DeviceRadixSort::SortPairs
    DeviceBuffer<int>   d_keys_in   = h_keys_in;
    DeviceBuffer<int> d_values_in = h_values_in;
    DeviceBuffer<int>   d_keys_out(size);
    DeviceBuffer<int> d_values_out(size);


    on().next<DeviceRadixSort>()
        .SortPairs(d_keys_in.data(),
                   d_keys_out.data(),
                   d_values_in.data(),
                   d_values_out.data(),
                   size)
        .wait();

    // Copy results from device to host
    d_keys_out.copy_to(h_keys_out);
    d_values_out.copy_to(h_values_out);
}
 

void device_radix_sort_sort_pairs_descending(std::vector<int>&   h_keys_out,
                                             std::vector<int>& h_values_out,
                                             std::vector<int>&   gt_keys_out,
                                             std::vector<int>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int>   h_keys_in(size);
    std::vector<int> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort in descending order
    gt_keys_out   = h_keys_in;
    gt_values_out = h_values_in;

    std::vector<Sortable> sortable(size);

    for(size_t i = 0; i < size; ++i)
    {
        sortable[i].id   = h_keys_in[i];
        sortable[i].data = h_values_in[i];
    }

    std::stable_sort(sortable.begin(), sortable.end(), std::greater<Sortable>());

    for(size_t i = 0; i < size; ++i)
    {
        gt_keys_out[i]   = sortable[i].id;
        gt_values_out[i] = sortable[i].data;
    }

    // Sort input data using DeviceRadixSort::SortPairsDescending
    DeviceBuffer<int>   d_keys_in   = h_keys_in;
    DeviceBuffer<int> d_values_in = h_values_in;
    DeviceBuffer<int>   d_keys_out(size);
    DeviceBuffer<int> d_values_out(size);


    on().next<DeviceRadixSort>()
        .SortPairsDescending(d_keys_in.data(),
                             d_keys_out.data(),
                             d_values_in.data(),
                             d_values_out.data(),
                             size)
        .wait();

    // Copy results from device to host


    d_keys_out.copy_to(h_keys_out);
    d_values_out.copy_to(h_values_out);
}


void device_radix_sort_sort_keys(std::vector<int>& h_keys_out, std::vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort
    gt_keys_out = h_keys_in;
    std::stable_sort(gt_keys_out.begin(), gt_keys_out.end());

    // Sort input data using DeviceRadixSort::SortKeys
    DeviceBuffer<int> d_keys_in = h_keys_in;
    DeviceBuffer<int> d_keys_out(size);


    on().next<DeviceRadixSort>()
        .SortKeys(d_keys_in.data(), d_keys_out.data(), size)
        .wait();

    // Copy results from device to host
    d_keys_out.copy_to(h_keys_out);
}


void device_radix_sort_sort_keys_descending(std::vector<int>& h_keys_out,
                                            std::vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort in descending order
    gt_keys_out = h_keys_in;
    std::stable_sort(gt_keys_out.begin(), gt_keys_out.end(), std::greater<int>());

    // Sort input data using DeviceRadixSort::SortKeysDescending
    DeviceBuffer<int> d_keys_in = h_keys_in;
    DeviceBuffer<int> d_keys_out(size);


    on().next<DeviceRadixSort>()
        .SortKeysDescending(d_keys_in.data(), d_keys_out.data(), size)
        .wait();

    // Copy results from device to host
    d_keys_out.copy_to(h_keys_out);
}

TEST_CASE("device_radix_sort", "[cub]")
{

    SECTION("SortPairsDescending")
    {
        std::vector<int>   h_keys_out;
        std::vector<int> h_values_out;
        std::vector<int>   gt_keys_out;
        std::vector<int> gt_values_out;

        device_radix_sort_sort_pairs_descending(h_keys_out, h_values_out, gt_keys_out, gt_values_out);

        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("SortPairs")
    {
        std::vector<int>   h_keys_out;
        std::vector<int> h_values_out;
        std::vector<int>   gt_keys_out;
        std::vector<int> gt_values_out;

        device_radix_sort_sort_pairs(h_keys_out, h_values_out, gt_keys_out, gt_values_out);
        // Check if the results are equal
        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("SortKeys")
    {
        std::vector<int> h_keys_out, gt_keys_out;
        device_radix_sort_sort_keys(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("SortKeysDescending")
    {
        std::vector<int> h_keys_out, gt_keys_out;
        device_radix_sort_sort_keys_descending(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }
}


void device_merge_sort_sort_pairs(std::vector<int>&   h_keys_out,
                                  std::vector<int>& h_values_out,
                                  std::vector<int>&   gt_keys_out,
                                  std::vector<int>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int>   h_keys_in(size);
    std::vector<int> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort
    gt_keys_out   = h_keys_in;
    gt_values_out = h_values_in;

    std::vector<Sortable> sortable(size);

    for(size_t i = 0; i < size; ++i)
    {
        sortable[i].id   = h_keys_in[i];
        sortable[i].data = h_values_in[i];
    }

    std::stable_sort(sortable.begin(), sortable.end());

    for(size_t i = 0; i < size; ++i)
    {
        gt_keys_out[i]   = sortable[i].id;
        gt_values_out[i] = sortable[i].data;
    }
    
    // Sort input data using DeviceMergeSort::SortPairs
    DeviceBuffer<int>   d_keys   = h_keys_in;
    DeviceBuffer<int> d_values = h_values_in;


    on().next<DeviceMergeSort>()
        .SortPairs(d_keys.data(),
                   d_values.data(),
                   size,
                   [] __host__ __device__(int l, int r) { return l < r; })
        .wait();

    // Copy results from device to host
    d_keys.copy_to(h_keys_out);
    d_values.copy_to(h_values_out);
}


void device_merge_sort_sort_pairs_copy(std::vector<int>&   h_keys_out,
                                       std::vector<int>& h_values_out,
                                       std::vector<int>&   gt_keys_out,
                                       std::vector<int>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int>   h_keys_in(size);
    std::vector<int> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort
    gt_keys_out   = h_keys_in;
    gt_values_out = h_values_in;
    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(),
              indices.end(),
              [&](size_t a, size_t b) { return h_keys_in[a] < h_keys_in[b]; });
    for(size_t i = 0; i < size; ++i)
    {
        gt_keys_out[i]   = h_keys_in[indices[i]];
        gt_values_out[i] = h_values_in[indices[i]];
    }

    // Sort input data using DeviceMergeSort::SortPairsCopy
    DeviceBuffer<int>   d_keys_in   = h_keys_in;
    DeviceBuffer<int> d_values_in = h_values_in;
    DeviceBuffer<int>   d_keys_out(size);
    DeviceBuffer<int> d_values_out(size);


    on().next<DeviceMergeSort>()
        .SortPairsCopy(d_keys_in.data(),
                       d_values_in.data(),
                       d_keys_out.data(),
                       d_values_out.data(),
                       size,
                       [] __host__ __device__(int l, int r) { return l < r; })
        .wait();

    // Copy results from device to host
    d_keys_out.copy_to(h_keys_out);
    d_values_out.copy_to(h_values_out);
}

void device_merge_sort_sort_keys(std::vector<int>& h_keys_out, std::vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort
    gt_keys_out = h_keys_in;
    std::stable_sort(gt_keys_out.begin(), gt_keys_out.end());

    // Sort input data using DeviceMergeSort::SortKeys
    DeviceBuffer<int> d_keys = h_keys_in;


    on().next<DeviceMergeSort>()
        .SortKeys(d_keys.data(),
                  size,
                  [] __host__ __device__(int l, int r) { return l < r; })
        .wait();

    // Copy results from device to host
    d_keys.copy_to(h_keys_out);
}

void device_merge_sort_sort_keys_copy(std::vector<int>& h_keys_out, std::vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort
    gt_keys_out = h_keys_in;
    std::stable_sort(gt_keys_out.begin(), gt_keys_out.end());

    // Sort input data using DeviceMergeSort::SortKeysCopy
    DeviceBuffer<int> d_keys_in = h_keys_in;
    DeviceBuffer<int> d_keys_out(size);


    on().next<DeviceMergeSort>()
        .SortKeysCopy(d_keys_in.data(),
                      d_keys_out.data(),
                      size,
                      [] __host__ __device__(int l, int r) { return l < r; })
        .wait();

    // Copy results from device to host
    d_keys_out.copy_to(h_keys_out);
}

void device_merge_sort_stable_sort_pairs(std::vector<int>&   h_keys_out,
                                         std::vector<int>& h_values_out,
                                         std::vector<int>&   gt_keys_out,
                                         std::vector<int>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int>   h_keys_in(size);
    std::vector<int> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort
    gt_keys_out   = h_keys_in;
    gt_values_out = h_values_in;
    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(),
                     indices.end(),
                     [&](size_t a, size_t b)
                     { return h_keys_in[a] < h_keys_in[b]; });
    for(size_t i = 0; i < size; ++i)
    {
        gt_keys_out[i]   = h_keys_in[indices[i]];
        gt_values_out[i] = h_values_in[indices[i]];
    }

    // Sort input data using DeviceMergeSort::StableSortPairs
    DeviceBuffer<int>   d_keys   = h_keys_in;
    DeviceBuffer<int> d_values = h_values_in;


    on().next<DeviceMergeSort>()
        .StableSortPairs(d_keys.data(),
                         d_values.data(),
                         size,
                         [] __host__ __device__(int l, int r) { return l < r; })
        .wait();

    // Copy results from device to host
    d_keys.copy_to(h_keys_out);
    d_values.copy_to(h_values_out);
}

void device_merge_sort_stable_sort_keys(std::vector<int>& h_keys_out,
                                        std::vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });

    // Sort input data using std::stable_sort
    gt_keys_out = h_keys_in;
    std::stable_sort(gt_keys_out.begin(), gt_keys_out.end());

    // Sort input data using DeviceMergeSort::StableSortKeys
    DeviceBuffer<int> d_keys = h_keys_in;


    on().next<DeviceMergeSort>()
        .StableSortKeys(d_keys.data(),
                        size,
                        [] __host__ __device__(int l, int r) { return l < r; })
        .wait();

    // Copy results from device to host
    d_keys.copy_to(h_keys_out);
}

TEST_CASE("device_merge_sort", "[cub]")
{
    SECTION("SortPairs")
    {
        std::vector<int>   h_keys_out;
        std::vector<int> h_values_out;
        std::vector<int>   gt_keys_out;
        std::vector<int> gt_values_out;

        device_merge_sort_sort_pairs(h_keys_out, h_values_out, gt_keys_out, gt_values_out);

        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("SortPairsCopy")
    {
        std::vector<int>   h_keys_out;
        std::vector<int> h_values_out;
        std::vector<int>   gt_keys_out;
        std::vector<int> gt_values_out;

        device_merge_sort_sort_pairs_copy(h_keys_out, h_values_out, gt_keys_out, gt_values_out);

        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("SortKeys")
    {
        std::vector<int> h_keys_out;
        std::vector<int> gt_keys_out;

        device_merge_sort_sort_keys(h_keys_out, gt_keys_out);

        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("SortKeysCopy")
    {
        std::vector<int> h_keys_out, gt_keys_out;
        device_merge_sort_sort_keys_copy(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("StableSortPairs")
    {
        std::vector<int>   h_keys_out;
        std::vector<int> h_values_out;
        std::vector<int>   gt_keys_out;
        std::vector<int> gt_values_out;

        device_merge_sort_stable_sort_pairs(h_keys_out, h_values_out, gt_keys_out, gt_values_out);

        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("StableSortKeys")
    {
        std::vector<int> h_keys_out, gt_keys_out;
        device_merge_sort_stable_sort_keys(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }
}

void device_select_flagged(std::vector<int>& h_keys_out, std::vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });

    // Generate flags
    std::vector<int> h_flags(size);
    std::for_each(
        h_flags.begin(), h_flags.end(), [](int& r) { r = std::rand() % 101 % 2; });

    // Filter input data using std::copy_if
    gt_keys_out.reserve(size);
    size_t idx = 0;
    std::copy_if(h_keys_in.begin(),
                 h_keys_in.end(),
                 std::back_inserter(gt_keys_out),
                 [&](auto key) { return h_flags[idx++]; });

    // Filter input data using DeviceSelect::Flagged
    DeviceBuffer<int> d_keys_in = h_keys_in;
    DeviceBuffer<int> d_flags   = h_flags;
    DeviceBuffer<int> d_keys_out(size);
    DeviceVar<int>    d_num_selected_out;


    on().next<DeviceSelect>()
        .Flagged(d_keys_in.data(),
                 d_flags.data(),
                 d_keys_out.data(),
                 d_num_selected_out.data(),
                 size)
        .wait();

    d_keys_out.resize(d_num_selected_out);
    // Copy results from device to host
    d_keys_out.copy_to(h_keys_out);
}


void device_select_if(std::vector<int>& h_keys_out, std::vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });

    // Filter input data using std::copy_if
    gt_keys_out.reserve(size);
    std::copy_if(h_keys_in.begin(),
                 h_keys_in.end(),
                 std::back_inserter(gt_keys_out),
                 [](int key) { return key % 2 == 0; });

    // Filter input data using DeviceSelect::If
    DeviceBuffer<int> d_keys_in = h_keys_in;
    DeviceBuffer<int> d_keys_out(size);
    DeviceVar<int>    d_num_selected_out;


    on().next<DeviceSelect>()
        .If(d_keys_in.data(),
            d_keys_out.data(),
            d_num_selected_out.data(),
            size,
            [] __host__ __device__(int key) { return key % 2 == 0; })
        .wait();

    d_keys_out.resize(d_num_selected_out);
    // Copy results from device to host
    d_keys_out.copy_to(h_keys_out);
}

void device_select_unique(std::vector<int>& h_keys_out, std::vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });

    // Filter input data using std::unique
    gt_keys_out.reserve(size);
    std::unique_copy(h_keys_in.begin(), h_keys_in.end(), std::back_inserter(gt_keys_out));

    // Filter input data using DeviceSelect::Unique
    DeviceBuffer<int> d_keys_in = h_keys_in;
    DeviceBuffer<int> d_keys_out(size);
    DeviceVar<int>    d_num_selected_out;


    on().next<DeviceSelect>()
        .Unique(d_keys_in.data(), d_keys_out.data(), d_num_selected_out.data(), size)
        .wait();

    d_keys_out.resize(d_num_selected_out);
    // Copy results from device to host
    d_keys_out.copy_to(h_keys_out);
}


TEST_CASE("device_select", "[cub]")
{
    SECTION("Flagged")
    {
        std::vector<int> h_keys_out, gt_keys_out;
        device_select_flagged(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("If")
    {
        std::vector<int> h_keys_out, gt_keys_out;
        device_select_if(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("Unique")
    {
        std::vector<int> h_keys_out, gt_keys_out;
        device_select_unique(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }
}

void device_partition_if(std::vector<int>& h_keys_out, std::vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    std::vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand() % 101; });

    // Partition input data using std::partition
    gt_keys_out   = h_keys_in;
    size_t select = 0;
    std::partition(gt_keys_out.begin(),
                   gt_keys_out.end(),
                   [&](int key)
                   {
                       if(key % 2 == 0)
                       {
                           ++select;
                           return true;
                       }
                       return false;
                   });
    gt_keys_out.resize(select);
    std::stable_sort(gt_keys_out.begin(), gt_keys_out.end());

    // Partition input data using DevicePartition::If
    DeviceBuffer<int> d_keys_in = h_keys_in;
    DeviceBuffer<int> d_keys_out(size);
    DeviceVar<int>    d_num_selected_out;


    on().next<DevicePartition>()
        .If(d_keys_in.data(),
            d_keys_out.data(),
            d_num_selected_out.data(),
            size,
            [] __host__ __device__(int key) { return key % 2 == 0; })
        .wait();

    d_keys_out.resize(d_num_selected_out);
    // Copy results from device to host
    d_keys_out.copy_to(h_keys_out);
    std::stable_sort(h_keys_out.begin(), h_keys_out.end());
}

TEST_CASE("device_partition", "[cub]")
{
    SECTION("If")
    {
        std::vector<int> h_keys_out, gt_keys_out;
        device_partition_if(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }
}
