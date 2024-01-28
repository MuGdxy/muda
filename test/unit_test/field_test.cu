#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/syntax_sugar.h>
#include <muda/ext/field.h>
#include <muda/ext/eigen.h>

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
    auto& vel     = builder.entry("velocity").vector3<float>();
    auto& force   = builder.entry("force").vector3<float>();
    auto& I       = builder.entry("inertia").matrix3x3<float>();
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
    auto builder = particle.builder(layout);
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
    vel.fill(Vector3f::UnitY());
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