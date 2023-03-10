#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>
#include <muda/blas/blas.h>

using namespace muda;

TEST_CASE("scal_axpy", "[blas]")
{
    stream      s;
    blasContext ctx(s);

    host_vector<float> h_x_buffer(16);
    for(size_t i = 0; i < h_x_buffer.size(); i++)
        h_x_buffer[i] = float(i);

    device_buffer<float> x_buffer(s);
    x_buffer.copy_from(h_x_buffer);

    dense_vec<float> x(x_buffer.data(), x_buffer.size());

    device_buffer<float> y_buffer(s);
    y_buffer.copy_from(h_x_buffer);

    dense_vec<float> y(y_buffer.data(), y_buffer.size());

    on(ctx)
        .scal(3.0f, x)     // x = 3 * x
        .axpy(2.0f, x, y)  // y = 2 * x + y
        .wait();           //wait

    host_vector<float> hy;
    y_buffer.copy_to(hy);
    launch::wait_stream(s);

    host_vector<float> ground_thruth;
    ground_thruth = h_x_buffer;
    for(auto& v : ground_thruth)
        v *= 7;
    REQUIRE(hy == ground_thruth);
}

TEST_CASE("nrm2 copy")
{
    stream      s;
    blasContext ctx(s);

    host_vector<float> h_x_buffer(16);
    for(size_t i = 0; i < h_x_buffer.size(); i++)
        h_x_buffer[i] = float(i);

    device_buffer<float> x_buffer(s);
    x_buffer.copy_from(h_x_buffer);
    device_buffer<float> y_buffer(s, x_buffer.size());

    auto x = make_dense_vec(x_buffer);
    auto y = make_dense_vec(y_buffer);

    float nrm;
    on(ctx).copy(x, y).nrm2(y, nrm).wait();

    host_vector<float> ground_thruth     = h_x_buffer;
    float              ground_thruth_res = 0.0f;
    for(size_t i = 0; i < ground_thruth.size(); i++)
        ground_thruth[i] *= ground_thruth[i];
    for(size_t i = 0; i < ground_thruth.size(); i++)
        ground_thruth_res += ground_thruth[i];
    ground_thruth_res = std::sqrt(ground_thruth_res);
    REQUIRE(ground_thruth_res == nrm);
}
