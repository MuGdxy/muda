#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/syntax_sugar.h>
#include <muda/ext/field.h>
#include <muda/ext/eigen.h>
#include <muda/cub/device/device_reduce.h>

using namespace muda;
using namespace Eigen;

void field_test1(FieldEntryLayout layout)
{
    Field field;
    auto& particle = field["particle"];
    float dt       = 0.01f;

    // build the field
    auto  builder = particle.builder(layout);
    auto& m       = builder.entry("mass").scalar<float>();
    auto& pos     = builder.entry("position").vector3<float>();

    auto& vel   = builder.entry("velocity").vector3<float>();
    auto& force = builder.entry("force").vector3<float>();
    auto& I     = builder.entry("inertia").matrix3x3<float>();
    builder.build();

    // set size of the particle attributes
    constexpr int N = 10;
    particle.resize(N);

    ParallelFor(256)
        .apply(N,
               [m   = m.viewer(),
                pos = pos.viewer(),
                vel = vel.viewer(),
                f   = force.viewer(),
                I   = I.viewer()] $(int i)
               {
                   m(i)   = 1.0f;
                   pos(i) = Vector3f::Ones();
                   vel(i) = Vector3f::Zero();
                   f(i)   = Vector3f{0.0f, -9.8f, 0.0f};

                   *I.data(i, 1, 0) = 1;
                   I(i) += Matrix3f::Ones();
                   f.name();
                   auto x = pos(i);
                   print("position = %f %f %f\n", x.x(), x.y(), x.z());
               })
        .wait();

    particle.resize(N * 13);

    ParallelFor(256)
        .apply(N,
               [m   = m.cviewer(),
                pos = pos.viewer(),
                vel = vel.viewer(),
                f   = force.cviewer(),
                I   = I.cviewer(),
                dt] $(int i)
               {
                   auto     x = pos(i);
                   auto     v = vel(i);
                   Vector3f a = f(i) / m(i);

                   v = v + a * dt;
                   x = x + v * dt;
                   print("position = %f %f %f\n", x.x(), x.y(), x.z());
                   print("innerta diag = %f %f %f\n", I(i)(0, 0), I(i)(1, 1), I(i)(2, 2));
                   print("innerta(1,0) = %f\n", I(i)(1, 0));
               })
        .wait();
}

void field_test2(FieldEntryLayout layout)
{
    Field field;
    auto& particle = field["particle"];
    float dt       = 0.01f;

    // build the field
    auto builder = particle.AoSoA();
    // auto  builder = particle.builder<FieldEntryLayout::SoA>();
    auto& m = builder.entry("mass").scalar<float>();
    auto& I = builder.entry("inertia").matrix3x3<float>();
    builder.build();

    // set size of the particle attributes
    constexpr int N = 10;
    particle.resize(N);

    Logger logger;
    ParallelFor(256)
        .apply(N,
               [m = m.viewer(), I = I.viewer(), logger = logger.viewer()] $(int i)
               {
                   m(i) = 1.0f;
                   //pos(i) = Vector3f::Ones();
                   *I.data(i, 0, 0) = 1;
                   *I.data(i, 1, 0) = 2;
                   *I.data(i, 2, 0) = 3;
                   *I.data(i, 0, 1) = 4;
                   *I.data(i, 1, 1) = 5;
                   *I.data(i, 2, 1) = 6;
                   *I.data(i, 0, 2) = 7;
                   *I.data(i, 1, 2) = 8;
                   *I.data(i, 2, 2) = 9;
                   logger << "i=" << i << "\n"
                          << "m=" << m(i) << "\n"
                          << "I=\n"
                          << I(i) << "\n";
               })
        .wait();
    logger.retrieve();

    particle.resize(N * 2);

    ParallelFor(256)
        .apply(N,
               [m = m.viewer(), I = I.cviewer(), logger = logger.viewer()] $(int i)
               {
                   logger << "i=" << i << "\n"
                          << "m=" << m(i) << "\n"
                          << "I=\n"
                          << I(i) << "\n";
               })
        .wait();
}


void field_example(FieldEntryLayout layout)
{
    using namespace muda;
    using namespace Eigen;

    Field field;
    // create a subfield called "particle"
    // any entry in this field has the same size
    auto& particle = field["particle"];
    float dt       = 0.01f;

    // build the field:
    // auto builder = particle.AoSoA(); // compile time layout
    auto builder = particle.builder(FieldEntryLayout::AoSoA);  // runtime layout
    auto& m      = builder.entry("mass").scalar<float>();
    auto& pos    = builder.entry("position").vector3<float>();
    auto& pos_old = builder.entry("position_old").vector3<float>();
    auto& vel     = builder.entry("velocity").vector3<float>();
    auto& force   = builder.entry("force").vector3<float>();
    // matrix is also supported, but in this example we don't use it
    auto& I = builder.entry("inertia").matrix3x3<float>();
    builder.build();  // finish building the field

    // set size of the particle attributes
    constexpr int N = 10;
    particle.resize(N);

    Logger logger;

    ParallelFor(256)
        .kernel_name("setup_vars")
        .apply(N,
               [logger = logger.viewer(),
                m      = m.viewer(),
                pos    = pos.viewer(),
                vel    = vel.viewer(),
                f      = force.viewer()] $(int i)
               {
                   m(i)   = 1.0f;
                   pos(i) = Vector3f::Ones();
                   vel(i) = Vector3f::Zero();
                   f(i)   = Vector3f{0.0f, -9.8f, 0.0f};

                   logger << "--------------------------------\n"
                          << "i=" << i << "\n"
                          << "m=" << m(i) << "\n"
                          << "pos=" << pos(i) << "\n"
                          << "vel=" << vel(i) << "\n"
                          << "f=" << f(i) << "\n";
               })
        .wait();

    logger.retrieve();

    // safe resize, the data will be copied to the new buffer.
    // here we just show the possibility
    // later we only work on the first N particles
    particle.resize(N * 2);

    ParallelFor(256)
        .kernel_name("integration")
        .apply(N,
               [logger = logger.viewer(),
                m      = m.cviewer(),
                pos    = pos.viewer(),
                vel    = vel.viewer(),
                f      = force.cviewer(),
                dt] $(int i)
               {
                   auto     x = pos(i);
                   auto     v = vel(i);
                   Vector3f a = f(i) / m(i);

                   v = v + a * dt;
                   x = x + v * dt;

                   logger << "--------------------------------\n"
                          << "i=" << i << "\n"
                          << "m=" << m(i) << "\n"
                          << "pos=" << pos(i) << "\n"
                          << "vel=" << vel(i) << "\n"
                          << "f=" << f(i) << "\n";
               })
        .wait();

    logger.retrieve();

    // copy between entry and host
    std::vector<Vector3f> positions;
    pos.copy_to(positions);
    pos.copy_from(positions);

    // copy between entries
    pos_old.copy_from(pos);

    // copy between buffer and entry
    DeviceBuffer<Vector3f> pos_buf;
    pos.copy_to(pos_buf);
    pos.copy_from(pos_buf);

    // using cub to do reduction on any format
    DeviceVar<float> total_mass;
    DeviceReduce().Sum(m.view(), total_mass.view(), m.count());
}

template <FieldEntryLayout Layout>
void field_test(FieldEntryLayout layout, int N)
{
    Field field;
    auto& particle = field["particle"];

    // build the field
    auto  builder = particle.template builder<Layout>(layout);
    auto& m       = builder.entry("mass").template scalar<float>();
    auto& pos     = builder.entry("position").template vector3<float>();
    auto& vel     = builder.entry("velocity").template vector3<float>();
    auto& I       = builder.entry("inertia").template matrix3x3<float>();
    builder.build();

    particle.resize(N);

    std::vector<float>    h_m(N);
    std::vector<Vector3f> h_pos(N);
    std::vector<Vector3f> h_vel(N);
    std::vector<Matrix3f> h_I(N);


    std::vector<float>    res_h_m(N);
    std::vector<Vector3f> res_h_pos(N);
    std::vector<Vector3f> res_h_vel(N);
    std::vector<Matrix3f> res_h_I(N);

    for(int i = 0; i < N; ++i)
    {
        h_m[i]   = 1.0f * i;
        h_pos[i] = 2.0f * Vector3f::Ones() * i;
        h_vel[i] = Vector3f::UnitY();
        h_I[i]   = 3.0f * Matrix3f::Ones() * i;
    }

    // test entry: copy from device buffer & copy from host
    m.copy_from(h_m);
    pos.copy_from(h_pos);

    // test entry: fill
    vel.view(0, N - 1).fill(Vector3f::UnitY());
    vel.view(N - 1, 1).fill(Vector3f::UnitY());
    I.copy_from(h_I);

    // test entry: copy to device buffer & copy to host
    m.copy_to(res_h_m);
    pos.copy_to(res_h_pos);
    vel.copy_to(res_h_vel);
    I.copy_to(res_h_I);

    REQUIRE(h_m == res_h_m);
    REQUIRE(h_pos == res_h_pos);
    REQUIRE(h_vel == res_h_vel);
    REQUIRE(h_I == res_h_I);

    // test field resize
    particle.resize(N * 2);

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(N,
               [N, m = m.viewer(), pos = pos.viewer(), vel = vel.viewer(), I = I.viewer()] $(int i)
               {
                   m(N + i)   = -1.0f * i;
                   pos(N + i) = -Vector3f::Ones() * i;
                   vel(N + i) = -Vector3f::UnitY();
                   I(N + i)   = -Matrix3f::Ones() * i;
               });


    h_m.resize(N * 2);
    h_pos.resize(N * 2);
    h_vel.resize(N * 2);
    h_I.resize(N * 2);

    for(int i = 0; i < N; ++i)
    {
        h_m[N + i]   = -1.0f * i;
        h_pos[N + i] = -Vector3f::Ones() * i;
        h_vel[N + i] = -Vector3f::UnitY();
        h_I[N + i]   = -Matrix3f::Ones() * i;
    }

    m.copy_to(res_h_m);
    pos.copy_to(res_h_pos);
    vel.copy_to(res_h_vel);
    I.copy_to(res_h_I);

    REQUIRE(h_m == res_h_m);
    REQUIRE(h_pos == res_h_pos);
    REQUIRE(h_vel == res_h_vel);
    REQUIRE(h_I == res_h_I);

    // test entry: entry entry copy
    pos.copy_from(vel);
    h_pos = h_vel;

    pos.copy_to(res_h_pos);
    REQUIRE(h_pos == res_h_pos);


    DeviceBuffer<Vector3f> temp(2 * N);

    particle.resize(100 * N);
    temp.fill(Vector3f::Zero());
    pos.view(0, 2 * N).copy_to(temp);
    temp.copy_to(res_h_pos);

    REQUIRE(h_pos == res_h_pos);

    particle.resize(7 * N);
    temp.fill(Vector3f::Zero());
    pos.view(0, 2 * N).copy_to(temp);
    temp.copy_to(res_h_pos);
    REQUIRE(h_pos == res_h_pos);
}

TEST_CASE("field_test", "[field]")
{
    using Layout = FieldEntryLayout;

    std::array layout{Layout::AoSoA, Layout::SoA, Layout::AoS};
    std::array name{"Runtime:AoSoA", "Runtime:SoA", "Runtime:AoS"};

    for(int i = 0; i < layout.size(); ++i)
    {
        SECTION(name[i])
        {
            field_test<Layout::RuntimeLayout>(layout[i], 10);
            field_test<Layout::RuntimeLayout>(layout[i], 33);
            field_test<Layout::RuntimeLayout>(layout[i], 197);
        }
    }

    SECTION("AoSoA")
    {
        field_test<Layout::AoSoA>(Layout::AoSoA, 10);
        field_test<Layout::AoSoA>(Layout::AoSoA, 33);
        field_test<Layout::AoSoA>(Layout::AoSoA, 197);
    }

    SECTION("SoA")
    {
        field_test<Layout::SoA>(Layout::SoA, 10);
        field_test<Layout::SoA>(Layout::SoA, 33);
        field_test<Layout::SoA>(Layout::SoA, 197);
    }

    SECTION("AoS")
    {
        field_test<Layout::AoS>(Layout::AoS, 10);
        field_test<Layout::AoS>(Layout::AoS, 33);
        field_test<Layout::AoS>(Layout::AoS, 197);
    }
}

#include <muda/cub/device/device_scan.h>

void field_cub(FieldEntryLayout input_layout, FieldEntryLayout output_layout, size_t size)
{
    Field field;
    auto& input  = field["input"];
    auto& output = field["output"];

    auto  builder1 = input.builder(input_layout);
    auto& x        = builder1.entry("x").scalar<int>();
    auto& vx       = builder1.entry("vx").vector3<float>();
    auto& mx       = builder1.entry("mx").matrix3x3<float>();
    builder1.build();

    auto  builder2 = output.builder(output_layout);
    auto& y        = builder2.entry("y").scalar<int>();
    auto& vy       = builder2.entry("vy").vector3<float>();
    auto& my       = builder2.entry("my").matrix3x3<float>();
    builder2.build();

    input.resize(size);
    output.resize(size);

    {  // test scalar
        std::vector<int> h_x(size, 1);
        std::vector<int> h_gt_y(size);
        std::exclusive_scan(h_x.begin(), h_x.end(), h_gt_y.begin(), 0);

        std::vector<int> h_y(size);
        x.copy_from(h_x);
        DeviceScan().ExclusiveSum(x.view(), y.view(), x.count());
        y.copy_to(h_y);

        REQUIRE(h_y == h_gt_y);
    }

    {  // test vector
        std::vector<Vector3f> h_vx(size, Vector3f::Ones());
        std::vector<Vector3f> h_vy(size, Vector3f::Ones());
        std::exclusive_scan(
            h_vx.begin(), h_vx.end(), h_vy.begin(), Vector3f::Zero().eval());

        std::vector<Vector3f> h_vy2(size);
        vx.copy_from(h_vx);
        DeviceScan().ExclusiveScan(
            vx.view(),
            vy.view(),
            [] __host__ __device__(const Vector3f& a, const Vector3f& b) -> Vector3f
            { return a + b; },
            Vector3f::Zero().eval(),
            vx.count());
        vy.copy_to(h_vy2);

        REQUIRE(h_vy == h_vy2);
    }

    {  // test matrix
        std::vector<Matrix3f> h_mx(size, Matrix3f::Ones());
        std::vector<Matrix3f> h_my(size, Matrix3f::Ones());
        std::exclusive_scan(
            h_mx.begin(), h_mx.end(), h_my.begin(), Matrix3f::Zero().eval());

        std::vector<Matrix3f> h_my2(size);
        mx.copy_from(h_mx);
        DeviceScan().ExclusiveScan(
            mx.view(),
            my.view(),
            [] __host__ __device__(const Matrix3f& a, const Matrix3f& b) -> Matrix3f
            { return a + b; },
            Matrix3f::Zero().eval(),
            mx.count());
        my.copy_to(h_my2);

        REQUIRE(h_my == h_my2);
    }
}

TEST_CASE("field_cub", "[field]")
{
    using Layout = FieldEntryLayout;

    std::array layout{Layout::AoSoA, Layout::SoA, Layout::AoS};
    std::array name{"Runtime:AoSoA", "Runtime:SoA", "Runtime:AoS"};

    for(int i = 0; i < layout.size(); ++i)
    {
        for(int j = 0; j < layout.size(); ++j)
        {
            SECTION(name[i] + std::string{"->"} + name[j])
            {
                field_cub(layout[i], layout[j], 10);
                field_cub(layout[i], layout[j], 33);
                field_cub(layout[i], layout[j], 197);
            }
        }
    }
}