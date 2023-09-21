#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>

using namespace muda;

void buffer_resize_test(int count, int new_count, BufferOperation op, HostVector<int>& result)
{
    Stream             s;
    int*               data;
    DeviceBuffer<int> buf(s, count);
    //set value
    on(s).next<ParallelFor>(32, 32).apply(buf.size(),
                                           [idx = make_viewer(buf)] __device__(int i) mutable
                                           { idx(i) = 1; });
    buf.resize(new_count, op);
    buf.copy_to(result).wait();
}

TEST_CASE("buffer_realloc_test", "[buffer]")
{
    HostVector<int> ground_thruth;
    HostVector<int> res;

    SECTION("expand_keep")
    {
        auto count     = 10;
        auto new_count = 2 * count;
        ground_thruth.resize(count, 1);
        HostVector<int> res;
        buffer_resize_test(count, new_count, BufferOperation::keep, res);
        res.resize(count);
        REQUIRE(ground_thruth == res);
    }

    SECTION("expand_set")
    {
        auto count     = 10;
        auto new_count = 2 * count;
        ground_thruth.resize(new_count, 0);
        HostVector<int> res;
        buffer_resize_test(count, new_count, BufferOperation::set, res);
        REQUIRE(ground_thruth == res);
    }

    SECTION("expand_keep_set")
    {
        auto count     = 10;
        auto new_count = 2 * count;
        ground_thruth.resize(new_count, 0);
        for(size_t i = 0; i < count; i++)
            ground_thruth[i] = 1;
        HostVector<int> res;
        buffer_resize_test(count, new_count, BufferOperation::keep_set, res);
        REQUIRE(ground_thruth == res);
    }

    SECTION("shrink_set")
    {
        auto count     = 20;
        auto new_count = count / 2;
        ground_thruth.resize(new_count, 0);
        HostVector<int> res;
        buffer_resize_test(count, new_count, BufferOperation::set, res);
        REQUIRE(ground_thruth == res);
    }

    SECTION("shrink_keep")
    {
        auto count     = 20;
        auto new_count = count / 2;
        ground_thruth.resize(new_count, 1);
        HostVector<int> res;
        buffer_resize_test(count, new_count, BufferOperation::keep, res);
        REQUIRE(ground_thruth == res);
    }
}

using vec3 = Eigen::Vector3f;

void buffer_resize_test(HostVector<vec3>& ground_thruth, HostVector<vec3>& res)
{
    Stream s;

    DeviceBuffer<vec3> buf(s);
    buf.resize(32, vec3::Ones());
    ground_thruth.resize(32, vec3::Ones());
    buf.copy_to(res).wait();
}

TEST_CASE("buffer_resize_test", "[buffer]")
{
    HostVector<vec3> ground_thruth, res;
    buffer_resize_test(ground_thruth, res);
    REQUIRE(ground_thruth == res);
}


TEST_CASE("buffer_launch_test", "[buffer]")
{

    SECTION("resize test")
    {
        Stream           s;
        HostVector<int> ground_thruth, res;
        ground_thruth.resize(8, 1);

        DeviceBuffer<int> buf;

        auto buf_launch = on(s).next<BufferOperator>();
        buf_launch.resize(buf, 8, 1).copy_to(buf, res).wait();
        REQUIRE(ground_thruth == res);

        buf_launch.resize(buf, 8).copy_to(buf, res).wait();
        ground_thruth.resize(8, 0);
        REQUIRE(ground_thruth == res);

        buf_launch.resize(buf, 8, 1)
            .resize(buf, 8, BufferOperation::set, 0)
            .copy_to(buf, res)
            .wait();
        ground_thruth.clear();
        ground_thruth.resize(8, 0);
        REQUIRE(ground_thruth == res);
    }

    SECTION("shrink test")
    {
        Stream           s;
        HostVector<int> ground_thruth, res;
        ground_thruth.resize(8, 1);

        DeviceBuffer<int> buf;

        on(s)
            .next<BufferOperator>()
            .copy_from(buf, ground_thruth)
            .resize(buf, 4)
            .shrink_to_fit(buf)
            .copy_to(buf, res)
            .wait();
        ground_thruth.resize(4, 1);
        REQUIRE(ground_thruth == res);
    }

    SECTION("copy test")
    {
        Stream s;
        {  // host -> buffer -> host
            HostVector<int> ground_thruth, res;
            ground_thruth.resize(8, 1);

            DeviceBuffer<int> buf;

            on(s)
                .next<BufferOperator>()
                .copy_from(buf, ground_thruth)
                .copy_to(buf, res)
                .wait();

            REQUIRE(ground_thruth == res);
        }

        {  // device -> buffer -> host
            HostVector<int> ground_thruth, res;
            ground_thruth.resize(8, 1);
            DeviceVector<int> device_data;
            device_data = ground_thruth;

            DeviceBuffer<int> buf;

            on(s)
                .next<BufferOperator>()
                .copy_from(buf, device_data)
                .copy_to(buf, res)
                .wait();

            REQUIRE(ground_thruth == res);
        }

        {  // host -> buffer -> device
            HostVector<int> ground_thruth, res;
            ground_thruth.resize(8, 1);
            DeviceVector<int> device_data;

            DeviceBuffer<int> buf;

            on(s)
                .next<BufferOperator>()
                .copy_from(buf, ground_thruth)
                .copy_to(buf, device_data)
                .wait();

            res = device_data;

            REQUIRE(ground_thruth == res);
        }

        {  // host -> buffer -> device
            HostVector<int> ground_thruth, res;
            ground_thruth.resize(8, 1);
            DeviceBuffer<int> buf2;
            buf2.resize(8, 1);

            DeviceBuffer<int> buf;

            on(s).next<BufferOperator>().copy_from(buf, buf2).copy_to(buf, res).wait();

            REQUIRE(ground_thruth == res);

            on(s).next<BufferOperator>().copy_to(buf, buf2).copy_to(buf2, res).wait();

            REQUIRE(ground_thruth == res);
        }
    }
}