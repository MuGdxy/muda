find_package(Eigen3 REQUIRED)

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found (include: ${EIGEN3_INCLUDE_DIR}, version: ${EIGEN3_VERSION_STRING})")
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

# find_package(fmt REQUIRED)
file(GLOB_RECURSE MUDA_EXAMPLE_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/example/*.cu"
    "${PROJECT_SOURCE_DIR}/example/*.cpp"
    "${PROJECT_SOURCE_DIR}/example/*.h")

add_executable(muda_example ${MUDA_EXAMPLE_SOURCE_FILES})
set_target_properties(muda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(muda_example PRIVATE fmt::fmt-header-only)
target_include_directories(muda_example PRIVATE
    "${PROJECT_SOURCE_DIR}/example"
    "${PROJECT_SOURCE_DIR}/external")
target_link_libraries(muda_example PRIVATE muda Eigen3::Eigen)
source_group(TREE "${PROJECT_SOURCE_DIR}/example" PREFIX "example" FILES ${MUDA_EXAMPLE_SOURCE_FILES})
source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "src" FILES ${MUDA_HEADER_FILES})
# set cuda sm75
set_property(TARGET muda_example PROPERTY CUDA_ARCHITECTURES 75)