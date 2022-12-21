#include "../example_common.h"
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/thread_only/priority_queue.h>

using namespace muda;
void priority_queue_example() 
{
    example_desc("use thread-only priority queue");
    launch(1, 1)
        .apply(
            [] __device__() mutable
            {
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
                print("queue_size = %d\n", queue.size());
                print("pop:");
                while(!queue.empty())
                {
                    print("%d ", queue.top());
                    queue.pop();
                }
                print("\n");
            })
        .wait();
}

TEST_CASE("priority_queue_example", "[thread_only]")
{
    priority_queue_example();
}