# project
cmake_minimum_required(VERSION 3.20)

# options
# include modules:
# option(MUDA_UTILS "includes <gui> <pba> modules." OFF)
# option(MUDA_EXT "includes <algorithm> <buffer> <blas> <composite> <thread-only> modules." ON)
# TODO: require GLFW
# option(MUDA_WITH_GUI "UNSPORTED when building with CMake" OFF)

# build targets:
option(MUDA_BUILD_EXAMPLE "build muda examples. if you want to see how to use muda, you could enable this option." ON)
option(MUDA_PLAYGROUND "build muda playground. if you're the developer, you could enable this option." OFF)
option(MUDA_BUILD_TEST "build muda test. if you're the developer, you could enable this option." OFF)

# short cut
option(MUDA_CORE_ONLY "only include the core functionality of muda." OFF)
option(MUDA_DEV "build muda example, playground and test. if you're the developer, you could enable this option." OFF)
option(NDEBUG "disable muda debug mode." OFF)

if(MUDA_CORE_ONLY)
    set(MUDA_DEV OFF)
    set(MUDA_BUILD_EXAMPLE OFF)
    set(MUDA_PLAYGROUND OFF)
    set(MUDA_BUILD_TEST OFF)
    
    # set(MUDA_WITH_GUI OFF)
    # set(MUDA_UTILS OFF)
    # set(MUDA_EXT OFF)
endif()

if(MUDA_DEV)
    set(MUDA_BUILD_EXAMPLE ON)
    set(MUDA_PLAYGROUND ON)
    set(MUDA_BUILD_TEST ON)
endif()

# to remove warning
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(muda LANGUAGES CXX CUDA)

find_package(CUDAToolkit)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)

# target

# muda core
add_library(muda-core INTERFACE)
target_compile_options(muda-core INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)
target_link_libraries(muda-core INTERFACE ${CUDA_LIBRARIES})
target_include_directories(muda-core INTERFACE "${PROJECT_SOURCE_DIR}/src/core")
target_include_directories(muda-core INTERFACE "${PROJECT_SOURCE_DIR}/external/default")

#if(MUDA_EXT)
add_library(muda-ext INTERFACE)
target_link_libraries(muda-ext INTERFACE muda-core)
target_include_directories(muda-ext INTERFACE "${PROJECT_SOURCE_DIR}/src/ext")
# muda-eastl config
target_compile_definitions(muda-ext INTERFACE "-D_CHAR16T")
target_compile_definitions(muda-ext INTERFACE "-D_CRT_SECURE_NO_WARNINGS")
target_compile_definitions(muda-ext INTERFACE "-D_SCL_SECURE_NO_WARNINGS")
target_compile_definitions(muda-ext INTERFACE "-DEASTL_OPENSOURCE=1")
target_include_directories(muda-ext INTERFACE "${PROJECT_SOURCE_DIR}/src/ext/muda/thread_only") # EASTL requirement
target_include_directories(muda-ext INTERFACE "${PROJECT_SOURCE_DIR}/src/ext/muda/thread_only/EABase/include/common") # EASTL requirement
#endif()

# if(MUDA_UTILS)
add_library(muda-pba INTERFACE)
target_link_libraries(muda-pba INTERFACE muda-core)
target_include_directories(muda-pba INTERFACE "${PROJECT_SOURCE_DIR}/src/util")
# endif()

add_library(muda INTERFACE)
target_link_libraries(muda INTERFACE muda-core muda-ext muda-pba)
file(GLOB_RECURSE MUDA_HEADER_FILES
  "${PROJECT_SOURCE_DIR}/src/*.h")

if(MUDA_BUILD_EXAMPLE)
  file(GLOB_RECURSE MUDA_EXAMPLE_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/example/*.cu"
  "${PROJECT_SOURCE_DIR}/example/*.cpp"
  "${PROJECT_SOURCE_DIR}/example/*.h")
  
  add_executable(muda_example ${MUDA_EXAMPLE_SOURCE_FILES})
  set_target_properties(muda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  target_include_directories(muda_example PRIVATE 
    "${PROJECT_SOURCE_DIR}/example"
    "${PROJECT_SOURCE_DIR}/test"
    )
  target_link_libraries(muda_example PRIVATE muda)
  source_group(TREE "${PROJECT_SOURCE_DIR}/example" PREFIX "example" FILES ${MUDA_EXAMPLE_SOURCE_FILES})
  source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "src" FILES ${MUDA_HEADER_FILES})
endif()

if(MUDA_BUILD_TEST)
  file(GLOB_RECURSE MUDA_TEST_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/test/muda_test/*.cu"
  "${PROJECT_SOURCE_DIR}/test/muda_test/*.cpp"
  "${PROJECT_SOURCE_DIR}/test/muda_test/*.h")
  add_executable(muda_test ${MUDA_TEST_SOURCE_FILES})
  set_target_properties(muda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(muda_test PRIVATE 
    "${PROJECT_SOURCE_DIR}/test"
    )
  target_link_libraries(muda_test PRIVATE muda cusparse cublas)
  target_compile_definitions(muda_test PRIVATE "-DMUDA_TEST_DATA_DIR=R\"(${PROJECT_SOURCE_DIR}/test/data)\"")
  source_group(TREE "${PROJECT_SOURCE_DIR}/test" PREFIX "test" FILES ${MUDA_TEST_SOURCE_FILES})
  source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "src" FILES ${MUDA_HEADER_FILES})
endif()