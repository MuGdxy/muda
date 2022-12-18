#include <catch2/catch.hpp>
#include <muda/muda.h>

using namespace muda;

#include <muda/algorithm/prefix_sum.h>
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
    SECTION("prefix_sum")
    {
        int              count = 99;
        host_vector<int> input(count, 1);
        host_vector<int> gt_ex(count, 0);
        host_vector<int> gt_in(count, 0);
        for(size_t i = 0; i < count; i++)
        {
            if(i > 0)
            {
                gt_ex[i] = gt_ex[i - 1] + input[i - 1];
                gt_in[i] = gt_in[i - 1] + input[i];
            }
            else
            {
                gt_ex[i] = 0;
                gt_in[i] = input[i];
            }
        }
        host_vector<int> ex(count, 1);
        host_vector<int> in(count, 1);
        prefix_sum(input, ex, in);
        REQUIRE(ex == gt_ex);
        REQUIRE(in == gt_in);
    }
}

#include <muda/algorithm/radix_sort.h>
//radix sort
void radix_sort(host_vector<int>& key_in,
                host_vector<int>& key_out,
                host_vector<int>& value_in,
                host_vector<int>& value_out)
{
    size_t             count  = key_in.size();
    device_vector<int> keyin  = key_in;
    device_vector<int> keyout = key_out;
    device_buffer      buf;
    RadixSort().SortKeys(buf, data(keyout), data(keyin), count).wait();
}