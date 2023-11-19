#include <muda/tools/version.h>

#ifdef MUDA_BASELINE_CUDACC_VER_SATISFIED
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>
#include <muda/cuda/cooperative_groups.h>
#include <muda/cuda/cooperative_groups/memcpy_async.h>



using namespace muda;
namespace cg = cooperative_groups;
void async_transfer(HostVector<int>& res, HostVector<int>& ground_thruth)
{
    DeviceVector<int> data(128, 1);
    Launch(2, 64)
        .apply(
            [data = data.viewer()] __device__() mutable
            {
                __shared__ int smem[64];
                auto           block = cg::this_thread_block();
                cg::memcpy_async(block,
                                 smem,
                                 &data(block.group_index().x * block.num_threads()),
                                 64 * sizeof(int));
                cg::wait(block);
                int gtid   = cg::this_grid().thread_rank();
                int btid   = block.thread_rank();
                smem[btid] = gtid;
                block.sync();
                cg::memcpy_async(block,
                                 &data(block.group_index().x * block.num_threads()),
                                 smem,
                                 64 * sizeof(int));
                cg::wait(block);
            })
        .wait();
    ground_thruth.resize(128, 1);
    for(size_t i = 0; i < 128; i++)
        ground_thruth[i] = i;
    res = data;
}

TEST_CASE("async_transfer", "[cooperative_groups]")
{
    HostVector<int> res, ground_thruth;
    async_transfer(res, ground_thruth);
    REQUIRE(res == ground_thruth);
};
#endif