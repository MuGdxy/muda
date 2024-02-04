#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>
#include <Eigen/Core>
using namespace muda;

template <typename T>
std::vector<T> visualize(const DeviceBuffer<T>& buffer, std::string_view sep = " ")
{

    std::vector<T> h_res;
    buffer.copy_to(h_res);
    auto dense = make_dense_1d(h_res.data(), h_res.size());

    for(int i = 0; i < dense.dim(); ++i)
    {
        std::cout << dense(i) << sep;
    }
    std::cout << std::endl;
    return h_res;
}
template <typename T>
std::vector<T> visualize(const DeviceBuffer2D<T>& buffer, std::string_view sep = " ")
{
    std::vector<T> h_res;
    buffer.copy_to(h_res);
    auto dense =
        make_dense_2d(h_res.data(), buffer.extent().height(), buffer.extent().width());

    for(int i = 0; i < dense.dim().x; ++i)
    {
        for(int j = 0; j < dense.dim().y; ++j)
        {
            std::cout << dense(i, j) << sep;
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    return h_res;
}
template <typename T>
std::vector<T> visualize(const DeviceBuffer3D<T>& buffer, std::string_view sep = " ")
{
    std::vector<T> h_res;
    buffer.copy_to(h_res);
    auto dense = make_dense_3d(h_res.data(),
                               buffer.extent().depth(),
                               buffer.extent().height(),
                               buffer.extent().width());

    for(int k = 0; k < dense.dim().x; ++k)
    {
        for(int i = 0; i < dense.dim().y; ++i)
        {
            for(int j = 0; j < dense.dim().z; ++j)
            {
                std::cout << dense(k, i, j) << sep;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    return h_res;
}

struct TestStruct
{
    int          a;
    MUDA_GENERIC TestStruct() { a = 1; }
    MUDA_GENERIC ~TestStruct() { a = -1; }
    MUDA_GENERIC bool operator==(const TestStruct& rhs) const
    {
        return a == rhs.a;
    }
};

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

    SECTION("non-trivial")
    {
        REQUIRE(std::is_trivially_constructible_v<TestStruct> == false);
        REQUIRE(std::is_trivially_destructible_v<TestStruct> == false);

        DeviceBuffer<TestStruct> buffer{};
        DeviceBuffer<TestStruct> buffer_dst{};
        std::vector<TestStruct>  gt;

        REQUIRE(buffer.size() == gt.size());

        buffer.resize(77, TestStruct{});
        gt.resize(77, TestStruct{});
        REQUIRE(buffer.size() == gt.size());

        std::vector<TestStruct> h_res;

        buffer.copy_to(h_res);
        REQUIRE(h_res == gt);

        buffer.resize(99, TestStruct{});
        gt.resize(99, TestStruct{});
        REQUIRE(buffer.size() == gt.size());

        buffer.fill(TestStruct{});
        gt.assign(gt.size(), TestStruct{});

        buffer.copy_to(h_res);
        REQUIRE(h_res == gt);

        buffer.clear();
        gt.clear();
        REQUIRE(buffer.size() == gt.size());

        buffer.shrink_to_fit();
        gt.shrink_to_fit();
        REQUIRE(buffer.size() == gt.size());

        buffer.resize(100, TestStruct{});
        buffer_dst = buffer;
        gt.resize(100, TestStruct{});

        buffer_dst.copy_to(h_res);
        REQUIRE(h_res == gt);

        gt.clear();
        gt.resize(35, TestStruct{});
        buffer = gt;
        buffer.copy_to(h_res);
        REQUIRE(h_res == gt);
    }

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
        var        = buffer_var.view().as_const();  // via buffer view
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

TEST_CASE("buffer_2d_test", "[buffer]")
{
    SECTION("simple")
    {
        DeviceBuffer2D<int> buffer{};
        buffer.resize(Extent2D{137, 137}, 1);
        std::vector<int> gt(137 * 137, 1);
        std::vector<int> h_res;

        REQUIRE(buffer.view().total_size() == gt.size());


        buffer.copy_to(h_res);

        REQUIRE(h_res == gt);

        buffer.resize(Extent2D{99, 99}, 2);
        gt.resize(99 * 99, 2);
        buffer.copy_to(h_res);
        REQUIRE(h_res == gt);

        buffer.shrink_to_fit();
        gt.shrink_to_fit();

        DeviceBuffer2D<int> buffer2{buffer};
        buffer2.copy_to(h_res);
        REQUIRE(h_res == gt);

        buffer2.resize(Extent2D{122, 122}, 3);
        buffer2.copy_to(h_res);
        auto dense2d = make_dense_2d(h_res.data(), 122, 122);
        // inner boundary check
        REQUIRE(dense2d(98, 98) == 1);
        REQUIRE(dense2d(98, 97) == 1);
        REQUIRE(dense2d(97, 98) == 1);

        // outer boundary check
        REQUIRE(dense2d(98, 99) == 3);
        REQUIRE(dense2d(99, 98) == 3);
        REQUIRE(dense2d(99, 99) == 3);
    }

    SECTION("vis")
    {
        DeviceBuffer2D<int> buffer;
        buffer.resize(Extent2D{5, 5}, 1);
        std::cout << "resize: 5x5 with 1" << std::endl;
        auto res = visualize(buffer);
        std::cout << "resize: 7x2 with 2" << std::endl;
        buffer.resize(Extent2D{7, 2}, 2);

        {
            auto res = visualize(buffer);
            REQUIRE(std::all_of(
                res.begin(), res.end(), [](int v) { return v == 1 || v == 2; }));
        }

        std::cout << "resize: 2x7 with 3" << std::endl;
        buffer.resize(Extent2D{2, 7}, 3);
        {
            // because we trim the height, so the values of 2 are gone
            auto res = visualize(buffer);
            REQUIRE(std::all_of(
                res.begin(), res.end(), [](int v) { return v == 1 || v == 3; }));
        }

        std::cout << "resize: 9x9 with 4" << std::endl;
        buffer.resize(Extent2D{9, 9}, 4);
        {
            auto res = visualize(buffer);
            REQUIRE(std::all_of(res.begin(),
                                res.end(),
                                [](int v) { return v == 1 || v == 3 || v == 4; }));
        }
    }
}


TEST_CASE("buffer_3d_test", "[buffer]")
{
    DeviceBuffer3D<int> buffer{};
    buffer.resize(Extent3D{137, 137, 137}, 1);
    std::vector<int> gt(137 * 137 * 137, 1);
    std::vector<int> h_res;

    REQUIRE(buffer.view().total_size() == gt.size());

    buffer.copy_to(h_res);

    REQUIRE(h_res == gt);

    buffer.resize(Extent3D{99, 99, 99}, 2);
    gt.resize(99 * 99 * 99, 2);
    buffer.copy_to(h_res);
    REQUIRE(h_res == gt);

    buffer.shrink_to_fit();
    gt.shrink_to_fit();

    DeviceBuffer3D<int> buffer2 = buffer;
    buffer2.copy_to(h_res);
    REQUIRE(h_res == gt);
    REQUIRE(std::all_of(h_res.begin(), h_res.end(), [](int v) { return v == 1; }));

    buffer2.resize(Extent3D{122, 122, 122}, 3);
    buffer2.copy_to(h_res);
    auto dense3d = make_dense_3d(h_res.data(), 122, 122, 122);

    REQUIRE(std::all_of(
        h_res.begin(), h_res.end(), [](int v) { return v == 1 || v == 3; }));

    REQUIRE(dense3d(0, 0, 0) == 1);

    // inner boundary check
    REQUIRE(dense3d(98, 98, 98) == 1);
    REQUIRE(dense3d(98, 97, 98) == 1);
    REQUIRE(dense3d(97, 98, 98) == 1);

    // outer boundary check
    REQUIRE(dense3d(98, 99, 98) == 3);
    REQUIRE(dense3d(99, 98, 98) == 3);
    REQUIRE(dense3d(98, 98, 99) == 3);
    REQUIRE(dense3d(99, 99, 98) == 3);
    REQUIRE(dense3d(98, 99, 99) == 3);
    REQUIRE(dense3d(99, 98, 99) == 3);
    REQUIRE(dense3d(99, 99, 99) == 3);
}