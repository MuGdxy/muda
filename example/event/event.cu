#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>

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


    Stream         s1, s2;
    Event          set_value_done;
    DeviceVar<int> v = 1;
    on(s1)
        .next<Launch>(1, 1)
        .apply(
            [v = v.viewer()] __device__() mutable
            {
                int next = 2;
                MUDA_KERNEL_PRINT("kernel A on stream 1, set v = %d -> %d", v, next);
                v = next;
            })
        .record(set_value_done)
        .apply(
            [] __device__()
            {
                some_work();
                MUDA_KERNEL_PRINT("kernel B on stream 1 costs a lot of time");
            });

    on(s2)
        .when(set_value_done)
        .next<Launch>(1, 1)
        .apply([v = v.viewer()] __device__()
               { MUDA_KERNEL_PRINT("kernel C on stream 2, get v = %d", v); });

    wait_device();
}

TEST_CASE("event", "[quick_start]")
{
    event_record_and_wait();
}
