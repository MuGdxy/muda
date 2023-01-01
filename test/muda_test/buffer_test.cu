#include <catch2/catch.hpp>
#include <muda/muda.h>

using namespace muda;

void buffer_resize_test(int count, int new_count, buf_op op, host_vector<int>& result)
{
    muda::stream;
    stream             s;
    int*               data;
    device_buffer<int> buf(s, count);
    //set value
    on(s).next<parallel_for>(32, 32).apply(buf.size(),
                                           [idx = make_viewer(buf)] __device__(int i) mutable
                                           { idx(i) = 1; });
    buf.resize(new_count, op);
    buf.copy_to(result).wait();
}

TEST_CASE("buffer_realloc_test", "[buffer]")
{
    host_vector<int> gt;
    host_vector<int> res;

    SECTION("expand_keep")
    {
        auto count     = 10;
        auto new_count = 2 * count;
        gt.resize(count, 1);
        host_vector<int> res;
        buffer_resize_test(count, new_count, buf_op::keep, res);
        res.resize(count);
        REQUIRE(gt == res);
    }

    SECTION("expand_set")
    {
        auto count     = 10;
        auto new_count = 2 * count;
        gt.resize(new_count, 0);
        host_vector<int> res;
        buffer_resize_test(count, new_count, buf_op::set, res);
        REQUIRE(gt == res);
    }

    SECTION("expand_keep_set")
    {
        auto count     = 10;
        auto new_count = 2 * count;
        gt.resize(new_count, 0);
        for(size_t i = 0; i < count; i++)
            gt[i] = 1;
        host_vector<int> res;
        buffer_resize_test(count, new_count, buf_op::keep_set, res);
        REQUIRE(gt == res);
    }

    SECTION("shrink_set")
    {
        auto count     = 20;
        auto new_count = count / 2;
        gt.resize(new_count, 0);
        host_vector<int> res;
        buffer_resize_test(count, new_count, buf_op::set, res);
        REQUIRE(gt == res);
    }

    SECTION("shrink_keep")
    {
        auto count     = 20;
        auto new_count = count / 2;
        gt.resize(new_count, 1);
        host_vector<int> res;
        buffer_resize_test(count, new_count, buf_op::keep, res);
        REQUIRE(gt == res);
    }
}

using vec3 = Eigen::Vector3f;

void buffer_resize_test(host_vector<vec3>& gt, host_vector<vec3>& res)
{
    stream s;
    
    device_buffer<vec3> buf(s);
    buf.resize(32, vec3::Ones());
    gt.resize(32, vec3::Ones());
    buf.copy_to(res).wait();
}

TEST_CASE("buffer_resize_test", "[buffer]") 
{
    host_vector<vec3> gt, res;
    buffer_resize_test(gt, res);
    REQUIRE(gt == res);
}
