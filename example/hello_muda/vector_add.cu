#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <numeric>
#include <algorithm>
#include "../example_common.h"
using namespace muda;

void vector_add(host_vector<float>& gt_C, host_vector<float>& C)
{
    example_desc(
        R"(This is a well known vector_add example:
We will do C = A + B, where A B C are all vectors, like the cuda-sample:
https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAdd/vectorAdd.cu
but in muda style.)");

    constexpr int        N = 1024;
    host_vector<float>   hA(N), hB(N);
    device_vector<float> dA(N), dB(N), dC(N);

    // initialize A and B using random numbers
    auto rand = [] { return std::rand() / (float)RAND_MAX; };
    std::generate(hA.begin(), hA.end(), rand);
    std::generate(hB.begin(), hB.end(), rand);

    // copy A and B to device
    dA = hA;
    dB = hB;

    // use grid-stride loop to cover all elements
    parallel_for(2, 256)
        .apply(N,
               [dC = make_viewer(dC),  // | this is a capture list              |
                dA = make_viewer(dA),  // | map from device_vector to a viewer  |
                dB = make_viewer(dB)]  // | which is the most muda-style part!  |
               __device__(int i) mutable  // place "mutable" to make dC modifiable
               {
                   // safe parallel_for will cover the rang [0, N)
                   // i just goes from 0 to N-1
                   dC(i) = dA(i) + dB(i);
               })
        .wait();  // wait the kernel to finish

    // copy C back to host
    C = dC;

    // do C = A + B in host
    gt_C.resize(N);
    std::transform(hA.begin(), hA.end(), hB.begin(), gt_C.begin(), std::plus<float>());
}

TEST_CASE("vector_add", "[quick start]")
{
    host_vector<float> gt_C, C;
    vector_add(gt_C, C);
    REQUIRE(gt_C == C);
}
