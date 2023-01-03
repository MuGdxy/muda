#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include "../example_common.h"

using namespace muda;

void stream_async()
{
    example_desc(
        "use async streams to launch kernels, and set some call backs\n"
        "to be executed when the kernels are done.");

    std::array<stream, 2> streams;

    std::cout << "launch kernel A on stream 1" << std::endl;
    universal_var<int> sum(0);
    on(streams[0])
        .next(launch(1, 1))
        .apply(
            [sum = make_viewer(sum)] __device__() mutable
            {
                for(int i = 0; i < 1 << 20; i++)
                    sum /= i;
                print("[kernel print] kernel A on stream 1 costs a lot of time\n");
            })
        .callback([] __host__(cudaStream_t stream, cudaError error)
                  { std::cout << "stream 1 callback" << std::endl; });

    std::cout << "launch kernel B on stream 2" << std::endl;
    on(streams[1])
        .next(launch(1, 1))
        .apply(
            [] __device__() {
                print("[kernel print] kernel B on stream 2 costs a little bit time\n");
            })
        .next(host_call())
        .apply([] __host__()
               { std::cout << "host_call after kernel B" << std::endl; })
        .callback([] __host__(cudaStream_t stream, cudaError error)
                  { std::cout << "stream 2 callback" << std::endl; });

    launch::wait_device();

    std::cout << "launch kernel D" << std::endl;
    parallel_for(1).apply(1,
                          [] __device__(int i)
                          {
                              int sum = 1;
                              for(int i = 0; i < 1 << 20; i++)
                                  sum /= i;
                              print("[kernel print] kernel D on stream 0 costs a lot of time\n");
                          });
    std::cout << "after kernel D (no sync)" << std::endl;
    launch::wait_device();
    std::cout << "after kernel D (device sync)" << std::endl;
}

TEST_CASE("stream_async", "[quick_start]")
{
    stream_async();
}