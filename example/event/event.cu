#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include "../example_common.h"

using namespace muda;

void event_record_and_wait()
{
    example_desc(R"(use event to synchronize between streams:
        stream1       stream2
          A              
          |              
        record         when 
        event0 ------ event0
          |             |
          B             C)");


    stream          s1, s2;
    event           set_value_done;
    device_var<int> v = 1;
    on(s1)
        .next<launch>(1, 1)
        .apply(
            [v = make_viewer(v)] __device__() mutable
            {
                int next = 2;
                print("kernel A on stream 1, set v = %d -> %d\n", v, next);
                v = next;
            })
        .record(set_value_done)
        .apply(
            [] __device__()
            {
                some_work();
                print("kernel B on stream 1 costs a lot of time\n");
            });

    on(s2)
        .when(set_value_done)
        .next<launch>(1, 1)
        .apply([v = make_viewer(v)] __device__()
               { print("kernel C on stream 2, get v = %d\n", v); });
    launch::wait_device();
}

TEST_CASE("event", "[quick_start]")
{
    event_record_and_wait();
}
