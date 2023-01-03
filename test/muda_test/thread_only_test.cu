#include <catch2/catch.hpp>
#include <type_traits>
#include <numeric>
#include <vector>
#include <algorithm>

#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>

#include <muda/thread_only/vector.h>
#include <muda/thread_only/algorithm.h>
#include <muda/thread_only/numeric.h>
using namespace muda;


struct vector_test_result
{
    int acc;
    int max_diff;
    int min_diff;
    int inner_product;
    int sum_of_partial_sum;
    int sum_of_adjacent_difference;
};

void vector_test(vector_test_result& ground_thruth_result, vector_test_result& result)
{
    host_call()
        .apply(
            [&result = ground_thruth_result] __host__()
            {
                namespace to = std;
                to::vector<int> v(10);
                to::iota(v.begin(), v.end(), 0);
                result.acc      = to::accumulate(v.begin(), v.end(), 0);
                auto max        = to::max_element(v.begin(), v.end());
                result.max_diff = to::distance(v.begin(), max);
                auto min        = to::max_element(v.begin(), v.end());
                result.min_diff = to::distance(v.begin(), min);
                result.inner_product =
                    to::inner_product(v.begin(), v.end(), v.begin(), 0);
                to::vector<int> part(v.size());
                to::partial_sum(v.begin(), v.end(), part.begin());
                result.sum_of_partial_sum = to::accumulate(part.begin(), part.end(), 0);
                to::vector<int> adj(v.size());
                to::adjacent_difference(v.begin(), v.end(), adj.begin());
                result.sum_of_adjacent_difference =
                    to::accumulate(adj.begin(), adj.end(), 0);
            })
        .wait();

    device_var<vector_test_result> res;
    launch(1, 1)
        .apply(
            [r = make_viewer(res)] __device__() mutable
            {
                vector_test_result& result = r;
                namespace to               = muda::thread_only;
                to::vector<int> v(10);
                to::iota(v.begin(), v.end(), 0);
                result.acc      = to::accumulate(v.begin(), v.end(), 0);
                auto max        = to::max_element(v.begin(), v.end());
                result.max_diff = to::distance(v.begin(), max);
                auto min        = to::max_element(v.begin(), v.end());
                result.min_diff = to::distance(v.begin(), min);
                result.inner_product =
                    to::inner_product(v.begin(), v.end(), v.begin(), 0);
                to::vector<int> part(v.size());
                to::partial_sum(v.begin(), v.end(), part.begin());
                result.sum_of_partial_sum = to::accumulate(part.begin(), part.end(), 0);
                to::vector<int> adj(v.size());
                to::adjacent_difference(v.begin(), v.end(), adj.begin());
                result.sum_of_adjacent_difference =
                    to::accumulate(adj.begin(), adj.end(), 0);
            })
        .wait();
    result = res;
}

TEST_CASE("vector", "[thread_only]")
{
    vector_test_result ground_thruth, result;
    vector_test(ground_thruth, result);
    CHECK(ground_thruth.acc == result.acc);
    CHECK(ground_thruth.min_diff == result.min_diff);
    CHECK(ground_thruth.max_diff == result.max_diff);
    CHECK(ground_thruth.inner_product == result.inner_product);
    CHECK(ground_thruth.sum_of_partial_sum == result.sum_of_partial_sum);
    CHECK(ground_thruth.sum_of_partial_sum == result.sum_of_partial_sum);
}

#include <queue>
#include <muda/thread_only/priority_queue.h>

struct priority_queue_test_result
{
    int sum_of_adjacent_difference;
};

void priority_queue_test(priority_queue_test_result& ground_thruth_result,
                         priority_queue_test_result& result)
{
    host_call()
        .apply(
            [&result = ground_thruth_result] __host__()
            {
                namespace to = std;
                to::priority_queue<int> queue;

                to::vector<int> container;

                queue.push(4);
                queue.push(5);
                queue.push(6);
                queue.push(7);
                queue.push(8);
                queue.push(9);

                while(queue.size() > 0)
                {
                    container.push_back(queue.top());
                    queue.pop();
                }

                to::vector<int> adj(container.size());
                to::adjacent_difference(container.begin(), container.end(), adj.begin());
                result.sum_of_adjacent_difference =
                    to::accumulate(container.begin(), container.end(), 0);
            })
        .wait();

    device_var<priority_queue_test_result> res;
    launch(1, 1)
        .apply(
            [r = make_viewer(res)] __device__() mutable
            {
                priority_queue_test_result& result = r;
                
                namespace to = muda::thread_only;

                to::priority_queue<int> queue;
                auto& container = queue.get_container();

                container.reserve(16);
                queue.push(4);
                queue.push(5);
                queue.push(6);
                queue.push(7);
                queue.push(8);
                queue.push(9);
                to::vector<int> adj(container.size());
                to::adjacent_difference(container.begin(), container.end(), adj.begin());
                result.sum_of_adjacent_difference =
                    to::accumulate(container.begin(), container.end(), 0);
            })
        .wait();
    result = res;
}

TEST_CASE("priority_queue", "[thread_only]")
{
    priority_queue_test_result ground_thruth, result;
    priority_queue_test(ground_thruth, result);
    CHECK(ground_thruth.sum_of_adjacent_difference == result.sum_of_adjacent_difference);
}