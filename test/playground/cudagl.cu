#include <catch2/catch.hpp>
#include <muda/gui/gui.h>

#include <stdio.h>

int CUDAGLRun(void)
{
    muda::IMuGuiMode  mode{muda::RaytraceMesh};
    muda::MuGuiCudaGL gui{mode};
    gui.init();
    while(!gui.frame())
    {
    }
    return EXIT_SUCCESS;
}

TEST_CASE("cuda_gl", "[gui]")
{
    REQUIRE(CUDAGLRun() == 0);
}