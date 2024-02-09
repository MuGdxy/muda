#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <thrust/device_vector.h>
#include <example_common.h>

void thrust_with_muda()
{
    example_desc(R"(In this example, we use 3 ways to setup a `thrust::device_vector`:
- pure thrust                                   [unsafe]
- `thrust::device_vector` with muda viewer      [safe but verbose]
- `muda::DeviceVector`                          [safe and concise])");

    using namespace muda;
    using namespace thrust;

    constexpr int N = 10;


    {
        device_vector<float> x(N);

        for_each(thrust::cuda::par_nosync,
                 make_counting_iterator(0),
                 make_counting_iterator(N),
                 [x = x.data()] __device__(int i) mutable
                 {
                     // unsafe access to x
                     x[i] = i;
                 });
    }

    {
        device_vector<float> x(N);

        KernelLabel lebal{"setup_verbose"};
        for_each(thrust::cuda::par_nosync,
                 make_counting_iterator(0),
                 make_counting_iterator(N),
                 [
                     // safe but too verbose
                     x = make_dense_1d(raw_pointer_cast(x.data()), N).name("x")] __device__(int i) mutable
                 {
                     // safe access to x, with range check
                     x(i) = i;
                 });
    }

    {
        DeviceVector<float> x(N);

        KernelLabel lebal{"setup_concise"};
        for_each(thrust::cuda::par_nosync,
                 make_counting_iterator(0),
                 make_counting_iterator(N),
                 [
                     // safe and concise
                     x = x.viewer().name("x")] __device__(int i) mutable
                 {
                     // safe access to x, with range check
                     x(i) = i;
                 });
    }
}

TEST_CASE("thrust", "[thrust]")
{
    thrust_with_muda();
}
