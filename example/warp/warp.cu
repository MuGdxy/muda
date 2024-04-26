#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <example_common.h>

#include <muda/cuda/cooperative_groups.h>
#include <muda/cuda/cooperative_groups/memcpy_async.h>
#include <muda/cuda/cooperative_groups/reduce.h>
#include <muda/cuda/cooperative_groups/scan.h>
#include <muda/cuda/device_atomic_functions.h>
#include <muda/cuda/cuda_runtime.h>
#include <muda/cuda/cuda_runtime_api.h>

using namespace muda;
namespace cg = cooperative_groups;
void warp()
{
    example_desc(R"(This is an example of warp level functions.
We use cooperative_groups to get warp level functions. 
For the sake of simplicity, we only launch one warp (even only 4 active thread).)");
    Launch(1, 4)
        .apply(
            [] __device__()
            {
                // get lane id using cooperative group
                auto block = cg::this_thread_block();
                auto this_warp = cg::tiled_partition<4>(block);
                int lane_id = this_warp.thread_rank();

                printf("[%d] lane_id: %d\n", lane_id, lane_id);

                auto res = cg::reduce(this_warp, lane_id, cg::greater<int>());
                printf("[%d] max(this_warp, lane_id): %d\n", lane_id, res);

                auto scan_res = cg::exclusive_scan(this_warp, 1);
                printf("[%d] scan(this_warp, 1): %d\n", lane_id, scan_res);

                auto result = this_warp.ballot(lane_id % 2);
                printf("[%d] ballot(lane_id %% 2): %x\n", lane_id, result);

                auto shlf_down_res = this_warp.shfl_down(lane_id, 1);
                printf("[%d] shfl_down(lane_id, delta=1): %d\n", lane_id, shlf_down_res);

                auto shlf_up_res = this_warp.shfl_up(lane_id, 1);
                printf("[%d] shfl_up(lane_id, delta=1): %d\n", lane_id, shlf_up_res);

                auto shlf_xor_res = this_warp.shfl_xor(lane_id, 0x3);
                printf("[%d] shfl_xor(lane_id, mask=0x3): %d\n", lane_id, shlf_xor_res);
            })
        .wait();
}

TEST_CASE("warp", "[default]")
{
    warp();
}
