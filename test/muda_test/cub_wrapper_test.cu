#include <numeric>
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/cub/cub.h>
#undef max
#undef min
using namespace muda;


struct Reducable
{
    int id = -1;
    int data;
};

bool operator==(const Reducable& lhs, const Reducable& rhs)
{
    return lhs.id == rhs.id;
}

void device_reduce_reduce(Reducable& h_output, Reducable& gt_output)
{
    device_buffer buffer;
    size_t        size = 100;

    host_vector<Reducable> gt_input(size);

    device_vector<Reducable> input;
    device_var<Reducable>    output;

    std::for_each(gt_input.begin(),
                  gt_input.end(),
                  [](Reducable& r) { r.id = std::rand(); });
    input = gt_input;

    on().next<DeviceReduce>()
        .Reduce(
            buffer,
            data(input),
            data(output),
            size,
            [] __device__(const Reducable& l, const Reducable& r) -> Reducable
            { return l.id > r.id ? l : r; },
            Reducable{})
        .wait();

    gt_output = *std::max_element(gt_input.begin(),
                                  gt_input.end(),
                                  [](const Reducable& l, const Reducable& r)
                                  { return l.id < r.id; });

    h_output = output;
}


void device_reduce_min(float& h_output, float& gt_output)
{
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float> gt_input(size);

    device_vector<float> input;
    device_var<float>    output;

    std::for_each(
        gt_input.begin(), gt_input.end(), [](float& r) { r = std::rand(); });
    input = gt_input;

    on().next<DeviceReduce>().Min(buffer, data(input), data(output), size).wait();

    gt_output = *std::min_element(gt_input.begin(), gt_input.end());

    h_output = output;
}

void device_reduce_max(float& h_output, float& gt_output)
{
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float> gt_input(size);

    device_vector<float> input;
    device_var<float>    output;

    std::for_each(
        gt_input.begin(), gt_input.end(), [](float& r) { r = std::rand(); });
    input = gt_input;

    on().next<DeviceReduce>().Max(buffer, data(input), data(output), size).wait();

    gt_output = *std::max_element(gt_input.begin(), gt_input.end());

    h_output = output;
}


void device_reduce_sum(float& h_output, float& gt_output)
{
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float> gt_input(size);

    device_vector<float> input;
    device_var<float>    output;

    std::for_each(
        gt_input.begin(), gt_input.end(), [](float& r) { r = std::rand(); });
    input = gt_input;

    on().next<DeviceReduce>().Sum(buffer, data(input), data(output), size).wait();

    gt_output = std::accumulate(gt_input.begin(), gt_input.end(), 0.0f);

    h_output = output;
}

void device_reduce_argmin(int& h_output, int& gt_output)
{
    using KVP = cub::KeyValuePair<int, float>;
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float>   gt_input(size);
    device_vector<float> input;
    device_var<KVP>      output;
    KVP                  h_output_kvp;


    std::for_each(
        gt_input.begin(), gt_input.end(), [](float& r) { r = std::rand(); });
    input = gt_input;

    // using std to get the index of the min element
    gt_output = std::min_element(gt_input.begin(), gt_input.end()) - gt_input.begin();


    on().next<DeviceReduce>().ArgMin(buffer, data(input), data(output), size).wait();

    h_output_kvp = output;
    h_output     = h_output_kvp.key;
}

void device_reduce_argmax(int& h_output, int& gt_output)
{
    using KVP = cub::KeyValuePair<int, float>;
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float>   gt_input(size);
    device_vector<float> input;
    device_var<KVP>      output;
    KVP                  h_output_kvp;

    std::for_each(
        gt_input.begin(), gt_input.end(), [](float& r) { r = std::rand(); });
    input = gt_input;

    // using std to get the index of the max element
    gt_output = std::max_element(gt_input.begin(), gt_input.end()) - gt_input.begin();


    on().next<DeviceReduce>().ArgMax(buffer, data(input), data(output), size).wait();

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

void device_reduce_reduce_by_key(host_vector<int>& h_unique_out,
                                 host_vector<int>& h_aggregates_out,
                                 int&              h_num_runs_out,
                                 host_vector<int>& gt_unique_out,
                                 host_vector<int>& gt_aggregates_out,
                                 int&              gt_num_runs_out)
{
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float>   gt_input(size);
    device_vector<float> input;

    // Declare, allocate, and initialize device-accessible pointers for input and output
    int num_items = 8;  // e.g., 8

    std::vector<int>   keys_in          = {0, 2, 2, 9, 5, 5, 5, 8};
    std::vector<int>   values_in        = {0, 7, 1, 6, 2, 5, 3, 4};
    device_vector<int> d_keys_in        = keys_in;
    device_vector<int> d_values_in      = values_in;
    device_vector<int> d_unique_out     = keys_in;
    device_vector<int> d_aggregates_out = keys_in;
    device_var<int>    d_num_runs_out;
    CustomMin          reduction_op;

    on().next<DeviceReduce>()
        .ReduceByKey(buffer,
                     data(d_keys_in),
                     data(d_unique_out),
                     data(d_values_in),
                     data(d_aggregates_out),
                     data(d_num_runs_out),
                     reduction_op,
                     num_items)
        .wait();

    d_unique_out.resize(d_num_runs_out);
    d_aggregates_out.resize(d_num_runs_out);

    h_unique_out     = d_unique_out;
    h_aggregates_out = d_aggregates_out;
    h_num_runs_out   = d_num_runs_out;

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
        float h_output;
        float gt_output;
        device_reduce_min(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("Max")
    {
        float h_output;
        float gt_output;
        device_reduce_max(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("Sum")
    {
        float h_output;
        float gt_output;
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
        host_vector<int> h_unique_out;
        host_vector<int> h_aggregates_out;
        int              h_num_runs_out;
        host_vector<int> gt_unique_out;
        host_vector<int> gt_aggregates_out;
        int              gt_num_runs_out;
        device_reduce_reduce_by_key(
            h_unique_out, h_aggregates_out, h_num_runs_out, gt_unique_out, gt_aggregates_out, gt_num_runs_out);
        REQUIRE(h_unique_out == gt_unique_out);
        REQUIRE(h_aggregates_out == gt_aggregates_out);
        REQUIRE(h_num_runs_out == gt_num_runs_out);
    }
}

void device_scan_inclusive_sum(host_vector<float>& h_output, host_vector<float>& gt_output)
{
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float> gt_input(size);
    gt_output.resize(size);
    device_vector<float> input(size);
    device_vector<float> output(size);

    std::for_each(
        gt_input.begin(), gt_input.end(), [](float& r) { r = std::rand(); });
    input = gt_input;

    // using std to get the inclusive sum
    std::partial_sum(gt_input.begin(), gt_input.end(), gt_output.begin());

    on().next<DeviceScan>()
        .InclusiveSum(buffer, data(input), data(output), size)
        .wait();

    h_output = output;
}

void device_scan_exclusive_sum(host_vector<float>& h_output, host_vector<float>& gt_output)
{
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float> gt_input(size);
    gt_output.resize(size);
    device_vector<float> input(size);
    device_vector<float> output(size);

    std::for_each(
        gt_input.begin(), gt_input.end(), [](float& r) { r = std::rand(); });
    input = gt_input;

    // using std to get the exclusive sum
    gt_output[0] = 0;
    std::partial_sum(gt_input.begin(), gt_input.end() - 1, gt_output.begin() + 1);

    on().next<DeviceScan>()
        .ExclusiveSum(buffer, data(input), data(output), size)
        .wait();

    h_output = output;
}

void device_scan_inclusive_scan(host_vector<float>& h_output, host_vector<float>& gt_output)
{
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float> gt_input(size);
    gt_output.resize(size);
    device_vector<float> input(size);
    device_vector<float> output(size);

    std::for_each(
        gt_input.begin(), gt_input.end(), [](float& r) { r = std::rand(); });
    input = gt_input;

    // using std to get the inclusive scan
    std::partial_sum(gt_input.begin(), gt_input.end(), gt_output.begin());

    on().next<DeviceScan>()
        .InclusiveScan(
            buffer,
            data(input),
            data(output),
            [] __device__(const float& a, const float& b) { return a + b; },
            size)
        .wait();

    h_output = output;
}

void device_scan_exclusive_scan(host_vector<float>& h_output, host_vector<float>& gt_output)
{
    device_buffer buffer;
    size_t        size = 100;

    host_vector<float> gt_input(size);
    gt_output.resize(size);
    device_vector<float> input(size);
    device_vector<float> output(size);

    std::for_each(
        gt_input.begin(), gt_input.end(), [](float& r) { r = std::rand(); });
    input = gt_input;

    // using std to get the exclusive scan
    gt_output[0] = 0;
    std::partial_sum(gt_input.begin(), gt_input.end() - 1, gt_output.begin() + 1);

    on().next<DeviceScan>()
        .ExclusiveScan(
            buffer,
            data(input),
            data(output),
            [] __device__(const float& a, const float& b) { return a + b; },
            0.0f,
            size)
        .wait();

    h_output = output;
}


void device_scan_exclusive_sum_by_key(host_vector<int>& h_values_out,
                                      host_vector<int>& gt_values_out)
{
    device_buffer buffer;
    size_t        size = 8;

    host_vector<int> h_keys_in   = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    host_vector<int> h_values_in = std::vector{0, 7, 1, 6, 2, 5, 3, 4};
    gt_values_out                = std::vector{0, 0, 7, 0, 0, 2, 7, 0};

    device_vector<int> d_keys_in   = h_keys_in;
    device_vector<int> d_values_in = h_values_in;
    device_vector<int> d_keys_out(size);
    device_vector<int> d_values_out(size);

    on().next<DeviceScan>()
        .ExclusiveSumByKey(buffer, data(d_keys_in), data(d_values_in), data(d_values_out), size)
        .wait();
    h_values_out = d_values_out;
}


void device_scan_inclusive_sum_by_key(host_vector<int>& h_values_out,
                                      host_vector<int>& gt_values_out)
{
    device_buffer buffer;
    size_t        size = 8;

    host_vector<int> h_keys_in   = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    host_vector<int> h_values_in = std::vector{0, 7, 1, 6, 2, 5, 3, 4};
    gt_values_out                = std::vector{0, 7, 8, 6, 2, 7, 10, 4};

    device_vector<int> d_keys_in   = h_keys_in;
    device_vector<int> d_values_in = h_values_in;
    device_vector<int> d_keys_out(size);
    device_vector<int> d_values_out(size);

    on().next<DeviceScan>()
        .InclusiveSumByKey(buffer, data(d_keys_in), data(d_values_in), data(d_values_out), size)
        .wait();

    h_values_out = d_values_out;
}


void device_scan_exclusive_scan_by_key(host_vector<int>& h_values_out,
                                       host_vector<int>& gt_values_out)
{
    device_buffer buffer;
    size_t        size = 8;

    host_vector<int> h_keys_in   = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    host_vector<int> h_values_in = std::vector{0, 7, 1, 6, 2, 5, 3, 4};
    gt_values_out                = std::vector{0, 0, 7, 0, 0, 2, 7, 0};

    device_vector<int> d_keys_in   = h_keys_in;
    device_vector<int> d_values_in = h_values_in;
    device_vector<int> d_keys_out(size);
    device_vector<int> d_values_out(size);

    on().next<DeviceScan>()
        .ExclusiveScanByKey(
            buffer,
            data(d_keys_in),
            data(d_values_in),
            data(d_values_out),
            [] __device__(const int& a, const int& b) -> int { return a + b; },
            0,
            size)
        .wait();

    h_values_out = d_values_out;
}


void device_scan_inclusive_scan_by_key(host_vector<int>& h_values_out,
                                       host_vector<int>& gt_values_out)
{
    device_buffer buffer;
    size_t        size = 8;

    host_vector<int> h_keys_in   = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    host_vector<int> h_values_in = std::vector{0, 7, 1, 6, 2, 5, 3, 4};
    gt_values_out                = std::vector{0, 7, 8, 6, 2, 7, 10, 4};

    device_vector<int> d_keys_in   = h_keys_in;
    device_vector<int> d_values_in = h_values_in;
    device_vector<int> d_keys_out(size);
    device_vector<int> d_values_out(size);


    on().next<DeviceScan>()
        .InclusiveScanByKey(
            buffer,
            data(d_keys_in),
            data(d_values_in),
            data(d_values_out),
            [] __device__(const int& a, const int& b) { return a + b; },
            size)
        .wait();

    h_values_out = d_values_out;
}

TEST_CASE("device_scan", "[cub]")
{
    SECTION("InclusiveSum")
    {
        host_vector<float> h_output, gt_output;
        device_scan_inclusive_sum(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("ExclusiveSum")
    {
        host_vector<float> h_output, gt_output;
        device_scan_exclusive_sum(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }


    SECTION("InclusiveScan")
    {
        host_vector<float> h_output, gt_output;
        device_scan_inclusive_scan(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }


    SECTION("ExclusiveScan")
    {
        host_vector<float> h_output, gt_output;
        device_scan_exclusive_scan(h_output, gt_output);
        REQUIRE(h_output == gt_output);
    }

    SECTION("ExclusiveSumByKey")
    {
        host_vector<int> h_values_out, gt_values_out;
        device_scan_exclusive_sum_by_key(h_values_out, gt_values_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("InclusiveSumByKey")
    {
        host_vector<int> h_values_out, gt_values_out;
        device_scan_inclusive_sum_by_key(h_values_out, gt_values_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("ExclusiveScanByKey")
    {
        host_vector<int> h_values_out, gt_values_out;
        device_scan_exclusive_scan_by_key(h_values_out, gt_values_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("InclusiveScanByKey")
    {
        host_vector<int> h_values_out, gt_values_out;
        device_scan_inclusive_scan_by_key(h_values_out, gt_values_out);
        REQUIRE(h_values_out == gt_values_out);
    }
}


void device_run_length_encode_encode(host_vector<int>& h_unique_out,
                                     host_vector<int>& h_counts_out,
                                     int&              h_num_runs_out,
                                     host_vector<int>& gt_unique_out,
                                     host_vector<int>& gt_counts_out,
                                     int&              gt_num_runs_out)
{
    device_buffer buffer;
    size_t        size = 8;

    host_vector<int> h_input = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    gt_unique_out            = std::vector{0, 2, 9, 5, 8};
    gt_counts_out            = std::vector{1, 2, 1, 3, 1};
    gt_num_runs_out          = 5;

    device_vector<int> d_input = h_input;
    device_vector<int> d_unique_out(size);
    device_vector<int> d_counts_out(size);
    device_var<int>    d_num_runs_out;

    on().next<DeviceRunLengthEncode>()
        .Encode(buffer, data(d_input), data(d_unique_out), data(d_counts_out), data(d_num_runs_out), size)
        .wait();

    d_unique_out.resize(d_num_runs_out);
    d_counts_out.resize(d_num_runs_out);

    h_unique_out   = d_unique_out;
    h_counts_out   = d_counts_out;
    h_num_runs_out = d_num_runs_out;
}

void device_run_length_encode_non_trivial_runs(host_vector<int>& h_offsets_out,
                                               host_vector<int>& h_counts_out,
                                               int&              h_num_runs_out,
                                               host_vector<int>& gt_offsets_out,
                                               host_vector<int>& gt_counts_out,
                                               int& gt_num_runs_out)
{
    device_buffer buffer;
    size_t        size = 8;

    host_vector<int> h_input = std::vector{0, 2, 2, 9, 5, 5, 5, 8};
    gt_offsets_out           = std::vector{1, 4};
    gt_counts_out            = std::vector{2, 3};
    gt_num_runs_out          = 2;

    device_vector<int> d_input = h_input;
    device_vector<int> d_offsets_out(size);
    device_vector<int> d_counts_out(size);
    device_var<int>    d_num_runs_out;

    on().next<DeviceRunLengthEncode>()
        .NonTrivialRuns(
            buffer, data(d_input), data(d_offsets_out), data(d_counts_out), data(d_num_runs_out), size)
        .wait();

    d_offsets_out.resize(d_num_runs_out);
    d_counts_out.resize(d_num_runs_out);

    h_offsets_out  = d_offsets_out;
    h_counts_out   = d_counts_out;
    h_num_runs_out = d_num_runs_out;
}

TEST_CASE("device_run_length_encode", "[cub]")
{
    SECTION("Encode")
    {
        host_vector<int> h_unique_out, h_counts_out, gt_unique_out, gt_counts_out;
        int h_num_runs_out, gt_num_runs_out;
        device_run_length_encode_encode(
            h_unique_out, h_counts_out, h_num_runs_out, gt_unique_out, gt_counts_out, gt_num_runs_out);
        REQUIRE(h_unique_out == gt_unique_out);
        REQUIRE(h_counts_out == gt_counts_out);
        REQUIRE(h_num_runs_out == gt_num_runs_out);
    }

    SECTION("NonTrivialRuns")
    {
        host_vector<int> h_offsets_out, h_counts_out, gt_offsets_out, gt_counts_out;
        int h_num_runs_out, gt_num_runs_out;
        device_run_length_encode_non_trivial_runs(
            h_offsets_out, h_counts_out, h_num_runs_out, gt_offsets_out, gt_counts_out, gt_num_runs_out);
        REQUIRE(h_offsets_out == gt_offsets_out);
        REQUIRE(h_counts_out == gt_counts_out);
        REQUIRE(h_num_runs_out == gt_num_runs_out);
    }
}

void device_radix_sort_sort_pairs(host_vector<int>&   h_keys_out,
                                  host_vector<float>& h_values_out,
                                  host_vector<int>&   gt_keys_out,
                                  host_vector<float>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int>   h_keys_in(size);
    host_vector<float> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](float& r) { r = std::rand(); });

    // Sort input data using std::sort
    gt_keys_out   = h_keys_in;
    gt_values_out = h_values_in;
    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(),
              indices.end(),
              [&](size_t a, size_t b) { return h_keys_in[a] < h_keys_in[b]; });
    for(size_t i = 0; i < size; ++i)
    {
        gt_keys_out[i]   = h_keys_in[indices[i]];
        gt_values_out[i] = h_values_in[indices[i]];
    }

    // Sort input data using DeviceRadixSort::SortPairs
    device_vector<int>   d_keys_in   = h_keys_in;
    device_vector<float> d_values_in = h_values_in;
    device_vector<int>   d_keys_out(size);
    device_vector<float> d_values_out(size);
    device_buffer        buffer;

    on().next<DeviceRadixSort>()
        .SortPairs(buffer, data(d_keys_in), data(d_keys_out), data(d_values_in), data(d_values_out), size)
        .wait();

    // Copy results from device to host
    h_keys_out   = d_keys_out;
    h_values_out = d_values_out;
}


void device_radix_sort_sort_pairs_descending(host_vector<int>&   h_keys_out,
                                             host_vector<float>& h_values_out,
                                             host_vector<int>&   gt_keys_out,
                                             host_vector<float>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int>   h_keys_in(size);
    host_vector<float> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](float& r) { r = std::rand(); });

    // Sort input data using std::sort in descending order
    gt_keys_out   = h_keys_in;
    gt_values_out = h_values_in;
    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(),
              indices.end(),
              [&](size_t a, size_t b) { return h_keys_in[a] > h_keys_in[b]; });
    for(size_t i = 0; i < size; ++i)
    {
        gt_keys_out[i]   = h_keys_in[indices[i]];
        gt_values_out[i] = h_values_in[indices[i]];
    }

    // Sort input data using DeviceRadixSort::SortPairsDescending
    device_vector<int>   d_keys_in   = h_keys_in;
    device_vector<float> d_values_in = h_values_in;
    device_vector<int>   d_keys_out(size);
    device_vector<float> d_values_out(size);
    device_buffer        buffer;

    on().next<DeviceRadixSort>()
        .SortPairsDescending(
            buffer, data(d_keys_in), data(d_keys_out), data(d_values_in), data(d_values_out), size)
        .wait();

    // Copy results from device to host
    h_keys_out   = d_keys_out;
    h_values_out = d_values_out;
}


void device_radix_sort_sort_keys(host_vector<int>& h_keys_out, host_vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });

    // Sort input data using std::sort
    gt_keys_out = h_keys_in;
    std::sort(gt_keys_out.begin(), gt_keys_out.end());

    // Sort input data using DeviceRadixSort::SortKeys
    device_vector<int> d_keys_in = h_keys_in;
    device_vector<int> d_keys_out(size);
    device_buffer      buffer;

    on().next<DeviceRadixSort>()
        .SortKeys(buffer, data(d_keys_in), data(d_keys_out), size)
        .wait();

    // Copy results from device to host
    h_keys_out = d_keys_out;
}


void device_radix_sort_sort_keys_descending(host_vector<int>& h_keys_out,
                                            host_vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });

    // Sort input data using std::sort in descending order
    gt_keys_out = h_keys_in;
    std::sort(gt_keys_out.begin(), gt_keys_out.end(), std::greater<int>());

    // Sort input data using DeviceRadixSort::SortKeysDescending
    device_vector<int> d_keys_in = h_keys_in;
    device_vector<int> d_keys_out(size);
    device_buffer      buffer;

    on().next<DeviceRadixSort>()
        .SortKeysDescending(buffer, data(d_keys_in), data(d_keys_out), size)
        .wait();

    // Copy results from device to host
    h_keys_out = d_keys_out;
}

TEST_CASE("device_radix_sort", "[cub]")
{

    SECTION("SortPairsDescending")
    {
        host_vector<int>   h_keys_out;
        host_vector<float> h_values_out;
        host_vector<int>   gt_keys_out;
        host_vector<float> gt_values_out;

        device_radix_sort_sort_pairs_descending(h_keys_out, h_values_out, gt_keys_out, gt_values_out);

        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("SortPairs")
    {
        host_vector<int>   h_keys_out;
        host_vector<float> h_values_out;
        host_vector<int>   gt_keys_out;
        host_vector<float> gt_values_out;

        device_radix_sort_sort_pairs(h_keys_out, h_values_out, gt_keys_out, gt_values_out);
        // Check if the results are equal
        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("SortKeys")
    {
        host_vector<int> h_keys_out, gt_keys_out;
        device_radix_sort_sort_keys(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("SortKeysDescending")
    {
        host_vector<int> h_keys_out, gt_keys_out;
        device_radix_sort_sort_keys_descending(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }
}


void device_merge_sort_sort_pairs(host_vector<int>&   h_keys_out,
                                  host_vector<float>& h_values_out,
                                  host_vector<int>&   gt_keys_out,
                                  host_vector<float>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int>   h_keys_in(size);
    host_vector<float> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](float& r) { r = std::rand(); });

    // Sort input data using std::sort
    gt_keys_out   = h_keys_in;
    gt_values_out = h_values_in;
    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(),
              indices.end(),
              [&](size_t a, size_t b) { return h_keys_in[a] < h_keys_in[b]; });
    for(size_t i = 0; i < size; ++i)
    {
        gt_keys_out[i]   = h_keys_in[indices[i]];
        gt_values_out[i] = h_values_in[indices[i]];
    }

    // Sort input data using DeviceMergeSort::SortPairs
    device_vector<int>   d_keys   = h_keys_in;
    device_vector<float> d_values = h_values_in;
    device_buffer        buffer;

    on().next<DeviceMergeSort>()
        .SortPairs(buffer,
                   data(d_keys),
                   data(d_values),
                   size,
                   [] __device__(auto l, auto r) { return l < r; })
        .wait();

    // Copy results from device to host
    h_keys_out   = d_keys;
    h_values_out = d_values;
}


void device_merge_sort_sort_pairs_copy(host_vector<int>&   h_keys_out,
                                       host_vector<float>& h_values_out,
                                       host_vector<int>&   gt_keys_out,
                                       host_vector<float>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int>   h_keys_in(size);
    host_vector<float> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](float& r) { r = std::rand(); });

    // Sort input data using std::sort
    gt_keys_out   = h_keys_in;
    gt_values_out = h_values_in;
    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(),
              indices.end(),
              [&](size_t a, size_t b) { return h_keys_in[a] < h_keys_in[b]; });
    for(size_t i = 0; i < size; ++i)
    {
        gt_keys_out[i]   = h_keys_in[indices[i]];
        gt_values_out[i] = h_values_in[indices[i]];
    }

    // Sort input data using DeviceMergeSort::SortPairsCopy
    device_vector<int>   d_keys_in   = h_keys_in;
    device_vector<float> d_values_in = h_values_in;
    device_vector<int>   d_keys_out(size);
    device_vector<float> d_values_out(size);
    device_buffer        buffer;

    on().next<DeviceMergeSort>()
        .SortPairsCopy(buffer,
                       data(d_keys_in),
                       data(d_values_in),
                       data(d_keys_out),
                       data(d_values_out),
                       size,
                       [] __device__(auto l, auto r) { return l < r; })
        .wait();

    // Copy results from device to host
    h_keys_out   = d_keys_out;
    h_values_out = d_values_out;
}

void device_merge_sort_sort_keys(host_vector<int>& h_keys_out, host_vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });

    // Sort input data using std::sort
    gt_keys_out = h_keys_in;
    std::sort(gt_keys_out.begin(), gt_keys_out.end());

    // Sort input data using DeviceMergeSort::SortKeys
    device_vector<int> d_keys = h_keys_in;
    device_buffer      buffer;

    on().next<DeviceMergeSort>()
        .SortKeys(buffer,
                  data(d_keys),
                  size,
                  [] __device__(auto l, auto r) { return l < r; })
        .wait();

    // Copy results from device to host
    h_keys_out = d_keys;
}

void device_merge_sort_sort_keys_copy(host_vector<int>& h_keys_out, host_vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });

    // Sort input data using std::sort
    gt_keys_out = h_keys_in;
    std::sort(gt_keys_out.begin(), gt_keys_out.end());

    // Sort input data using DeviceMergeSort::SortKeysCopy
    device_vector<int> d_keys_in = h_keys_in;
    device_vector<int> d_keys_out(size);
    device_buffer      buffer;

    on().next<DeviceMergeSort>()
        .SortKeysCopy(buffer,
                      data(d_keys_in),
                      data(d_keys_out),
                      size,
                      [] __device__(auto l, auto r) { return l < r; })
        .wait();

    // Copy results from device to host
    h_keys_out = d_keys_out;
}

void device_merge_sort_stable_sort_pairs(host_vector<int>&   h_keys_out,
                                         host_vector<float>& h_values_out,
                                         host_vector<int>&   gt_keys_out,
                                         host_vector<float>& gt_values_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int>   h_keys_in(size);
    host_vector<float> h_values_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });
    std::for_each(h_values_in.begin(),
                  h_values_in.end(),
                  [](float& r) { r = std::rand(); });

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
    device_vector<int>   d_keys   = h_keys_in;
    device_vector<float> d_values = h_values_in;
    device_buffer        buffer;

    on().next<DeviceMergeSort>()
        .StableSortPairs(buffer,
                         data(d_keys),
                         data(d_values),
                         size,
                         [] __device__(auto l, auto r) { return l < r; })
        .wait();

    // Copy results from device to host
    h_keys_out   = d_keys;
    h_values_out = d_values;
}

void device_merge_sort_stable_sort_keys(host_vector<int>& h_keys_out,
                                        host_vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });

    // Sort input data using std::stable_sort
    gt_keys_out = h_keys_in;
    std::stable_sort(gt_keys_out.begin(), gt_keys_out.end());

    // Sort input data using DeviceMergeSort::StableSortKeys
    device_vector<int> d_keys = h_keys_in;
    device_buffer      buffer;

    on().next<DeviceMergeSort>()
        .StableSortKeys(buffer,
                        data(d_keys),
                        size,
                        [] __device__(auto l, auto r) { return l < r; })
        .wait();

    // Copy results from device to host
    h_keys_out = d_keys;
}

TEST_CASE("device_merge_sort", "[cub]")
{
    SECTION("SortPairs")
    {
        host_vector<int>   h_keys_out;
        host_vector<float> h_values_out;
        host_vector<int>   gt_keys_out;
        host_vector<float> gt_values_out;

        device_merge_sort_sort_pairs(h_keys_out, h_values_out, gt_keys_out, gt_values_out);

        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("SortPairsCopy")
    {
        host_vector<int>   h_keys_out;
        host_vector<float> h_values_out;
        host_vector<int>   gt_keys_out;
        host_vector<float> gt_values_out;

        device_merge_sort_sort_pairs_copy(h_keys_out, h_values_out, gt_keys_out, gt_values_out);

        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("SortKeys")
    {
        host_vector<int> h_keys_out;
        host_vector<int> gt_keys_out;

        device_merge_sort_sort_keys(h_keys_out, gt_keys_out);

        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("SortKeysCopy")
    {
        host_vector<int> h_keys_out, gt_keys_out;
        device_merge_sort_sort_keys_copy(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("StableSortPairs")
    {
        host_vector<int>   h_keys_out;
        host_vector<float> h_values_out;
        host_vector<int>   gt_keys_out;
        host_vector<float> gt_values_out;

        device_merge_sort_stable_sort_pairs(h_keys_out, h_values_out, gt_keys_out, gt_values_out);

        REQUIRE(h_keys_out == gt_keys_out);
        REQUIRE(h_values_out == gt_values_out);
    }

    SECTION("StableSortKeys")
    {
        host_vector<int> h_keys_out, gt_keys_out;
        device_merge_sort_stable_sort_keys(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }
}

void device_select_flagged(host_vector<int>& h_keys_out, host_vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });

    // Generate flags
    host_vector<bool> h_flags(size);
    std::for_each(
        h_flags.begin(), h_flags.end(), [](bool& r) { r = std::rand() % 2; });

    // Filter input data using std::copy_if
    gt_keys_out.reserve(size);
    size_t idx = 0;
    std::copy_if(h_keys_in.begin(),
                 h_keys_in.end(),
                 std::back_inserter(gt_keys_out),
                 [&](auto key) { return h_flags[idx++]; });

    // Filter input data using DeviceSelect::Flagged
    device_vector<int>  d_keys_in = h_keys_in;
    device_vector<bool> d_flags   = h_flags;
    device_vector<int>  d_keys_out(size);
    device_var<int>     d_num_selected_out;
    device_buffer       buffer;

    on().next<DeviceSelect>()
        .Flagged(buffer, data(d_keys_in), data(d_flags), data(d_keys_out), data(d_num_selected_out), size)
        .wait();

    d_keys_out.resize(d_num_selected_out);
    // Copy results from device to host
    h_keys_out = d_keys_out;
}


void device_select_if(host_vector<int>& h_keys_out, host_vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });

    // Filter input data using std::copy_if
    gt_keys_out.reserve(size);
    std::copy_if(h_keys_in.begin(),
                 h_keys_in.end(),
                 std::back_inserter(gt_keys_out),
                 [](int key) { return key % 2 == 0; });

    // Filter input data using DeviceSelect::If
    device_vector<int> d_keys_in = h_keys_in;
    device_vector<int> d_keys_out(size);
    device_var<int>    d_num_selected_out;
    device_buffer      buffer;

    on().next<DeviceSelect>()
        .If(buffer,
            data(d_keys_in),
            data(d_keys_out),
            data(d_num_selected_out),
            size,
            [] __device__(auto key) { return key % 2 == 0; })
        .wait();

    d_keys_out.resize(d_num_selected_out);
    // Copy results from device to host
    h_keys_out = d_keys_out;
}

void device_select_unique(host_vector<int>& h_keys_out, host_vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });

    // Filter input data using std::unique
    gt_keys_out.reserve(size);
    std::unique_copy(h_keys_in.begin(), h_keys_in.end(), std::back_inserter(gt_keys_out));

    // Filter input data using DeviceSelect::Unique
    device_vector<int> d_keys_in = h_keys_in;
    device_vector<int> d_keys_out(size);
    device_var<int>    d_num_selected_out;
    device_buffer      buffer;

    on().next<DeviceSelect>()
        .Unique(buffer, data(d_keys_in), data(d_keys_out), data(d_num_selected_out), size)
        .wait();

    d_keys_out.resize(d_num_selected_out);
    // Copy results from device to host
    h_keys_out = d_keys_out;
}


TEST_CASE("device_select", "[cub]")
{
    SECTION("Flagged")
    {
        host_vector<int> h_keys_out, gt_keys_out;
        device_select_flagged(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("If")
    {
        host_vector<int> h_keys_out, gt_keys_out;
        device_select_if(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }

    SECTION("Unique")
    {
        host_vector<int> h_keys_out, gt_keys_out;
        device_select_unique(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }
}

void device_partition_if(host_vector<int>& h_keys_out, host_vector<int>& gt_keys_out)
{
    size_t size = 100;

    // Generate random input data
    host_vector<int> h_keys_in(size);
    std::for_each(
        h_keys_in.begin(), h_keys_in.end(), [](int& r) { r = std::rand(); });

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
    std::sort(gt_keys_out.begin(), gt_keys_out.end());

    // Partition input data using DevicePartition::If
    device_vector<int> d_keys_in = h_keys_in;
    device_vector<int> d_keys_out(size);
    device_var<int>    d_num_selected_out;
    device_buffer      buffer;

    on().next<DevicePartition>()
        .If(buffer,
            data(d_keys_in),
            data(d_keys_out),
            data(d_num_selected_out),
            size,
            [] __device__(auto key) { return key % 2 == 0; })
        .wait();

    d_keys_out.resize(d_num_selected_out);
    // Copy results from device to host
    h_keys_out = d_keys_out;
    std::sort(h_keys_out.begin(), h_keys_out.end());
}

TEST_CASE("device_partition", "[cub]")
{
    SECTION("If")
    {
        host_vector<int> h_keys_out, gt_keys_out;
        device_partition_if(h_keys_out, gt_keys_out);
        REQUIRE(h_keys_out == gt_keys_out);
    }
}
