#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>

using namespace muda;

void stream_async()
{
    example_desc(
        "use async streams to launch kernels, and set some call backs\n"
        "to be executed when the kernels are done.");

    std::array<Stream, 2> streams;

    std::cout << "launch kernel A on stream 1" << std::endl;
    on(streams[0])
        .next(Launch(1, 1))
        .apply(
            [] __device__() mutable
            {
                some_work();
                print("[kernel print] kernel A on stream 1 costs a lot of time\n");
            })
        .callback([] __host__(cudaStream_t stream, cudaError error)
                  { std::cout << "stream 1 callback\n"; });

    std::cout << "launch kernel B on stream 2\n";
    on(streams[1])
        .next(Launch(1, 1))
        .apply(
            [] __device__() {
                print("[kernel print] kernel B on stream 2 costs a little bit time\n");
            })
        .next(HostCall())
        .apply([] __host__() { std::cout << "host_call after kernel B\n"; })
        .callback([] __host__(cudaStream_t stream, cudaError error)
                  { std::cout << "stream 2 callback\n"; });

    wait_device();

    std::cout << "launch kernel D\n";
    ParallelFor(1).apply(1,
                          [] __device__(int i)
                          {
                              some_work();
                              print("[kernel print] kernel D on stream 0 costs a lot of time\n");
                          });
    std::cout << "after launch kernel D (no sync)\n";
    wait_device();
    std::cout << "after kernel D done (device sync)\n";
}

TEST_CASE("stream_async", "[quick_start]")
{
    stream_async();
}