/**
 * The roop simulation used to show how to extend the original gui for your own project
*/

#include <catch2/catch.hpp>
#include <muda/gui/gui.h>
#include <muda/muda.h>
#include <muda/buffer.h>

class RopeSim : public muda::MuGuiCudaGL
{
  public:
    RopeSim() = default;
    ~RopeSim() {}
    void muda_gen_vertices(float* positions, float time, unsigned int width, unsigned int height)
    {
        static float start_time = time;
        float        timestep   = 0.1;
        float        sim_time   = time - start_time;

        float left_ctr_x  = -0.4;
        float right_ctr_x = 0.4;
        float ctr_y_0     = 0.3;
        float rope_len  = sqrtf(right_ctr_x * right_ctr_x + ctr_y_0 * ctr_y_0);
        int   rope_segs = 100;
        float dx        = (right_ctr_x - left_ctr_x) / rope_segs;

        muda::device_vector<Eigen::Vector2f> rope(rope_segs);

        // update stage
        muda::parallel_for(rope_segs).apply(rope_segs,
                                            [time   = sim_time,
                                             dx     = dx,
                                             init_x = left_ctr_x,
                                             segs   = rope_segs,
                                             rope = muda::make_viewer(rope)] __device__(int i) mutable
                                            {
                                                float x = init_x + i * dx;
                                                float y = sinf(2.0f * x - time * 6.0f);
                                                // set the rope data with (x,y)

                                                // generate the current rope
                                                rope(i) = Eigen::Vector2f(x, y);
                                            });

        // visualize stage

        muda::parallel_for(rope_segs).apply(
            rope_segs,
            [rope  = muda::make_viewer(rope),
             width = width,
             positions = muda::make_dense3D(positions, height, width, 8)] __device__(int i) mutable
            {
                // read the rope by index i, then set to the position array (VBO)
                int ix              = (rope(i).x() + 1) / 2 * width;
                positions(0, ix, 0) = rope(i).x();
                positions(0, ix, 1) = rope(i).y();
                positions(0, ix, 2) = 0.0f;
                positions(0, ix, 3) = 1.0f;
                // generate color
                positions(0, ix, 4) = 0.5f;
                positions(0, ix, 5) = 0.3f;
                positions(0, ix, 6) = 0.8f;
                positions(0, ix, 7) = 1.0f;
            });
    }
};

int RopeSimRun(void)
{
    RopeSim sim{};
    sim.init();
    while(!sim.frame())
    {
    }
    return EXIT_SUCCESS;
}

TEST_CASE("rope_sim", "[gui]")
{
    REQUIRE(RopeSimRun() == 0);
}