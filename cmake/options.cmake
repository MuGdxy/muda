option(MUDA_PACKAGE "build muda as a package" OFF)
option(MUDA_FORCE_CHECK "turn on muda runtime check for all mode (Debug/RelWithDebInfo/Release)" OFF)
option(MUDA_WITH_CHECK "turn on muda runtime check when mode != Release" ON)
option(MUDA_WITH_COMPUTE_GRAPH "turn on muda compute graph" OFF)
option(MUDA_WITH_NVTX3 "turn on nividia tools extension library" OFF)

# build targets:
option(MUDA_BUILD_EXAMPLE "build muda examples. if you want to see how to use muda, you could enable this option." ON)
option(MUDA_BUILD_TEST "build muda test. if you're the developer, you could enable this option." OFF)
option(MUDA_APP_CXX_STANDARD "set c++ standard for muda app(test/example)" 20)

# short cut
option(MUDA_DEV "build muda example and unit test. if you're the developer, you could enable this option." OFF)

if(MUDA_DEV)
    set(MUDA_BUILD_EXAMPLE ON)
    set(MUDA_PLAYGROUND ON)
    set(MUDA_BUILD_TEST ON)
endif()