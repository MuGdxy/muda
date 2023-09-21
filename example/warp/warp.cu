#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include "../example_common.h"

#include <muda/cuda/cooperative_groups.h>
#include <muda/cuda/cooperative_groups/memcpy_async.h>
#include <muda/cuda/cooperative_groups/reduce.h>
#include <muda/cuda/cooperative_groups/scan.h>
#include <muda/cuda/device_atomic_functions.h>
#include <muda/cuda/device_functions.h>
#include <muda/cuda/cuda_runtime.h>
#include <muda/cuda/cuda_runtime_api.h>

using namespace muda;
namespace cg = cooperative_groups;
void warp()
{
    example_desc("warp");
    Launch(1, 16).apply(
        [] __device__()
        {
            // get lane id using cooperative group
            cg::thread_block block = cg::this_thread_block();
            cg::thread_block_tile<32> this_warp = cg::tiled_partition<32>(block);
            int lane_id = this_warp.thread_rank();

            printf("lane_id: %d\n", lane_id);

            auto res = cg::reduce(this_warp, lane_id, cg::greater<int>());
            printf("max: %d\n", res);

            auto scan_res = cg::inclusive_scan(this_warp, 1);
            printf("scan: %d\n", scan_res);

            auto result = this_warp.ballot(lane_id % 2);
            printf("ballot: %x\n", result);

            auto shlf_down_res = this_warp.shfl_down(lane_id, 1);
            printf("shfl_down: %d\n", shlf_down_res);

            auto shlf_up_res = this_warp.shfl_up(lane_id, 1);
            printf("shfl_up: %d\n", shlf_up_res);

            auto shlf_xor_res = this_warp.shfl_xor(lane_id, 1);
            printf("shfl_xor: %d\n", shlf_xor_res);
            double a;
            atomicAdd(&a, 1);
        });
}

TEST_CASE("warp", "[default]")
{
    warp();
}
