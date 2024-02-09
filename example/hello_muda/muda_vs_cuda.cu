#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>
using namespace muda;

__global__ void times2(int* i, int N)
{
    auto tid = threadIdx.x;
    if(tid < N)
    {
        i[tid] *= 2;
    }
}

void muda_vs_cuda()
{
    example_desc("muda vs cuda");

    // launch a kernel

    {  //muda
        DeviceVector<int> dv(64, 1);
        Stream             s;
        ParallelFor(1, 64, 0, s)
            .apply(64,
                   [dv = dv.viewer()] __device__(int i) mutable
                   { dv(i) *= 2; })
            .wait();
    }

    {  //cuda
        thrust::device_vector<int> dv(64, 1);
        auto                       dvptr = thrust::raw_pointer_cast(dv.data());
        cudaStream_t               s;
        checkCudaErrors(cudaStreamCreate(&s));
        times2<<<1, 64, 0, s>>>(dvptr, dv.size());
        checkCudaErrors(cudaStreamSynchronize(s));
    }
}

TEST_CASE("muda_vs_cuda", "[quick_start]")
{
    muda_vs_cuda();
}
