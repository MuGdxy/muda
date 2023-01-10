#include <catch2/catch.hpp>
#include <muda/gui/gui.h>

#include <stdio.h>

int PUREGLRun(void)
{
    muda::IMuGuiMode  mode{muda::RaytraceMesh};
    muda::MuGuiPureGL gui{mode};
    gui.init();
    while(!gui.frame())
    {
    }
    return EXIT_SUCCESS;
}

TEST_CASE("pure_gl", "[gui]")
{
    REQUIRE(PUREGLRun() == 0);
}