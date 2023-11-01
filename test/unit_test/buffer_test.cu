#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>
#include <Eigen/Core>
using namespace muda;

//struct TestStruct
//{
//    int          a;
//    MUDA_GENERIC TestStruct() { a = 1; }
//    MUDA_GENERIC ~TestStruct() { a = -1; }
//    MUDA_GENERIC bool operator==(const TestStruct& rhs) const
//    {
//        return a == rhs.a;
//    }
//};


TEST_CASE("buffer_test", "[buffer]")
{
    SECTION("trivial")
    {
        DeviceBuffer<int> buffer{};
        DeviceBuffer<int> buffer_dst{};
        std::vector<int>  gt;
        REQUIRE(buffer.size() == gt.size());
        REQUIRE(buffer.data() == nullptr);


        buffer.resize(77, 1);
        gt.resize(77, 1);
        REQUIRE(buffer.size() == gt.size());
        REQUIRE(buffer.data() != nullptr);

        std::vector<int> h_res;
        buffer.copy_to(h_res);
        REQUIRE(h_res == gt);

        buffer.resize(99, 2);
        gt.resize(99, 2);
        REQUIRE(buffer.size() == gt.size());
        buffer.copy_to(h_res);
        REQUIRE(h_res == gt);

        buffer.fill(3);
        gt.assign(gt.size(), 3);
        buffer.copy_to(h_res);
        REQUIRE(h_res == gt);

        buffer.clear();
        gt.clear();
        REQUIRE(buffer.size() == gt.size());

        buffer.shrink_to_fit();
        gt.shrink_to_fit();
        REQUIRE(buffer.size() == gt.size());
        REQUIRE(buffer.data() == nullptr);

        buffer.resize(100, 4);
        buffer_dst = buffer;
        gt.resize(100, 4);
        buffer_dst.copy_to(h_res);
        REQUIRE(h_res == gt);

        gt.clear();
        gt.resize(35, 5);
        buffer = gt;
        buffer.copy_to(h_res);
        REQUIRE(h_res == gt);
    }

    //SECTION("non-trivial")
    //{

    //    REQUIRE(std::is_trivially_constructible_v<TestStruct> == false);
    //    REQUIRE(std::is_trivially_destructible_v<TestStruct> == false);

    //    DeviceBuffer<TestStruct> buffer{};
    //    DeviceBuffer<TestStruct> buffer_dst{};
    //    std::vector<TestStruct>  gt;

    //    REQUIRE(buffer.size() == gt.size());

    //    buffer.resize(77, TestStruct{});
    //    gt.resize(77, TestStruct{});
    //    REQUIRE(buffer.size() == gt.size());

    //    std::vector<TestStruct> h_res;

    //    buffer.copy_to(h_res);
    //    REQUIRE(h_res == gt);

    //    buffer.resize(99, TestStruct{});
    //    gt.resize(99, TestStruct{});
    //    REQUIRE(buffer.size() == gt.size());

    //    buffer.fill(TestStruct{});
    //    gt.assign(gt.size(), TestStruct{});

    //    buffer.copy_to(h_res);
    //    REQUIRE(h_res == gt);

    //    buffer.clear();
    //    gt.clear();
    //    REQUIRE(buffer.size() == gt.size());

    //    buffer.shrink_to_fit();
    //    gt.shrink_to_fit();
    //    REQUIRE(buffer.size() == gt.size());

    //    buffer.resize(100, TestStruct{});
    //    buffer_dst = buffer;
    //    gt.resize(100, TestStruct{});

    //    buffer_dst.copy_to(h_res);
    //    REQUIRE(h_res == gt);

    //    gt.clear();
    //    gt.resize(35, TestStruct{});
    //    buffer = gt;
    //    buffer.copy_to(h_res);
    //    REQUIRE(h_res == gt);
    //}

    SECTION("buffer_view_test")
    {
        DeviceBuffer<float> buffer(10);
        DeviceVector<float> vec(10);
        buffer = vec;  // via buffer view
        REQUIRE(buffer.size() == vec.size());

        buffer = vec.view();  // via buffer view
        REQUIRE(buffer.size() == vec.size());

        vec = buffer;  // via buffer view

        DeviceVar<float> buffer_var;
        buffer_var = 1;
        DeviceVar<float> var;
        var = buffer_var;  // direct
        REQUIRE(var == buffer_var);

        buffer_var = 2;
        var        = buffer_var.view();  // via buffer view
        REQUIRE(var == buffer_var);
    }

    SECTION("buffer_eigen_vector_test")
    {
        using Vector12 = Eigen::Matrix<double, 12, 1>;

        DeviceBuffer<Vector12> buffer(10);
        DeviceBuffer<Vector12> buffer2(10);

        buffer2.fill(Vector12::Ones());
        buffer = buffer2;  // via buffer view
        std::vector<Vector12> gt(10, Vector12::Ones());

        std::vector<Vector12> h_res;
        buffer.copy_to(h_res);
        REQUIRE(h_res == gt);
    }
}