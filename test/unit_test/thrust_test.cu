#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/syntax_sugar.h>
#include <muda/ext/field.h>
#include <muda/ext/eigen.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/async/for_each.h>
#include <thrust/unique.h>

#include "../example/example_common.h"
#include <muda/launch/kernel_label.h>
#include <thrust/device_make_unique.h>

using namespace muda;
using namespace Eigen;

void thrust_test()
{
    using namespace thrust;
    // thrust::cuda::par.on(nullptr);  // set par stream
    auto          nosync_policy = thrust::cuda::par_nosync.on(nullptr);
    constexpr int N             = 100000;
    auto          ptr_n         = get_temporary_buffer<int>(nosync_policy, N);


    auto t0 = profile_host(
        [&]
        {
            for_each(nosync_policy,
                     make_counting_iterator(0),
                     make_counting_iterator(N),
                     [buffer = make_dense_1d(ptr_n.first.get(), N).name("buffer")] __device__(int i) mutable
                     {
                         buffer(i) = i;
                         some_work();
                     });
        });

    muda::wait_device();

    auto t_muda = profile_host(
        [&]
        {
            on(nullptr)  //
                .next<ParallelFor>()
                .kernel_name("muda")
                .apply(N,
                       [buffer = make_dense_1d(ptr_n.first.get(), N).name("buffer")] __device__(int i) mutable
                       {
                           buffer(i) = i;
                           // some_work();
                       });
        });

    std::cout << "muda launch time:" << t_muda << std::endl;

    muda::wait_device();

    auto t_thrust = profile_host(
        [&]
        {
            KernelLabel label{"thrust"};
            for_each(nosync_policy,
                     counting_iterator<int>{0},
                     counting_iterator<int>{N},
                     [buffer = make_dense_1d(ptr_n.first.get(), N).name("buffer")] __device__(int i) mutable
                     {
                         buffer(i) = i;
                         // some_work();
                     });
        });

    std::cout << "thrust launch time:" << t_thrust << std::endl;

    muda::wait_device();

    auto t_thrust_device = profile_host(
        [&]
        {
            KernelLabel label{"thrust"};
            for_each(nosync_policy,
                     counting_iterator<int>{0},
                     counting_iterator<int>{N},
                     [buffer = make_dense_1d(ptr_n.first.get(), N).name("buffer")] __device__(int i) mutable
                     {
                         buffer(i) = i;
                         // some_work();
                     });
        });
    std::cout << "thrust launch time:" << t_thrust_device << std::endl;
}

void pure_thrust()
{
    using namespace muda;
    using namespace thrust;

    auto nosync_policy = thrust::cuda::par_nosync.on(nullptr);

    constexpr auto N = 1000;

    thrust::device_vector<int> buffer(N);

    for_each(nosync_policy,
             thrust::make_counting_iterator(0),
             thrust::make_counting_iterator(N),
             [buffer = buffer.data()] __device__(int i) mutable
             {
                 // do some work
                 buffer[i] = i;
             });
    checkCudaErrors(cudaStreamSynchronize(nullptr));
}

void muda_thrust()
{
    using namespace muda;
    using namespace thrust;

    auto nosync_policy = thrust::cuda::par_nosync.on(nullptr);

    constexpr auto N = 1000;

    DeviceVector<int> buffer(N);

    {
        KernelLabel label{__FUNCTION__};

        for_each(nosync_policy,
                 thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(N),
                 [buffer = buffer.viewer().name("buffer")] __device__(int i) mutable
                 {
                     // do some work
                     buffer(i) = i;
                 });
    }
}


TEST_CASE("thrust_test", "[thrust]")
{
    thrust_test();
    muda_thrust();
    pure_thrust();
}
