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
option(MUDA_BUILD_UNIT_TEST "build muda test. if you're the developer, you could enable this option." OFF)

# short cut
option(MUDA_DEV "build muda example and unit test. if you're the developer, you could enable this option." OFF)
option(MUDA_WITH_CHECK "turn on muda runtime check" ON)



if(MUDA_DEV)
    set(MUDA_BUILD_EXAMPLE ON)
    set(MUDA_PLAYGROUND ON)
    set(MUDA_BUILD_UNIT_TEST ON)
endif()

# to remove warning
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(muda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)

# muda
add_library(muda INTERFACE)
target_compile_options(muda INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)
file(GLOB_RECURSE MUDA_HEADER_FILES "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/*.inl")

target_sources(muda PUBLIC ${MUDA_HEADER_FILES})
target_link_libraries(muda INTERFACE Eigen3::Eigen)
target_link_libraries(muda INTERFACE ${CUDA_LIBRARIES})
target_include_directories(muda INTERFACE "${PROJECT_SOURCE_DIR}/src/")
if(MUDA_WITH_CHECK)
  target_compile_definitions(muda INTERFACE "-DMUDA_CHECK_ON=1")
endif()

if(MUDA_BUILD_EXAMPLE)
  find_package(Eigen3 REQUIRED)
  if(Eigen3_FOUND)
    message(STATUS "Eigen3 found (include: ${EIGEN3_INCLUDE_DIR}, version: ${EIGEN3_VERSION_STRING})")
  else()
    message(FATAL_ERROR "Eigen3 not found")
  endif()
  #find_package(fmt REQUIRED)
  file(GLOB_RECURSE MUDA_EXAMPLE_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/example/*.cu"
  "${PROJECT_SOURCE_DIR}/example/*.cpp"
  "${PROJECT_SOURCE_DIR}/example/*.h")
  
  add_executable(muda_example ${MUDA_EXAMPLE_SOURCE_FILES})
  set_target_properties(muda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  #target_link_libraries(muda_example PRIVATE fmt::fmt-header-only)

  target_include_directories(muda_example PRIVATE 
    "${PROJECT_SOURCE_DIR}/example"
    "${PROJECT_SOURCE_DIR}/external")
  target_link_libraries(muda_example PRIVATE muda)
  source_group(TREE "${PROJECT_SOURCE_DIR}/example" PREFIX "example" FILES ${MUDA_EXAMPLE_SOURCE_FILES})
  source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "src" FILES ${MUDA_HEADER_FILES})
endif()

if(MUDA_BUILD_UNIT_TEST)
  file(GLOB_RECURSE MUDA_UNIT_TEST_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/test/unit_test/*.cu"
  "${PROJECT_SOURCE_DIR}/test/unit_test/*.cpp"
  "${PROJECT_SOURCE_DIR}/test/unit_test/*.h")
  add_executable(muda_test ${MUDA_UNIT_TEST_SOURCE_FILES})
  set_target_properties(muda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  target_include_directories(muda_test PRIVATE 
    "${PROJECT_SOURCE_DIR}/test"
    "${PROJECT_SOURCE_DIR}/external")
  target_link_libraries(muda_test PRIVATE muda cusparse cublas)
  target_compile_definitions(muda_test PRIVATE "-DMUDA_TEST_DATA_DIR=R\"(${PROJECT_SOURCE_DIR}/test/data)\"")
  source_group(TREE "${PROJECT_SOURCE_DIR}/test" PREFIX "test" FILES ${MUDA_UNIT_TEST_SOURCE_FILES})
  source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "src" FILES ${MUDA_HEADER_FILES})
endif()