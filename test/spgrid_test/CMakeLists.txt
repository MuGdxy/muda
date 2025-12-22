# spgrid test
find_package(Eigen3 REQUIRED)
file(GLOB_RECURSE SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_executable(muda_spgrid_test)
target_sources(muda_spgrid_test PRIVATE ${SOURCE} ${MUDA_HEADER_FILES})
target_compile_features(muda_spgrid_test PRIVATE cxx_std_20)
set_target_properties(muda_spgrid_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(muda_spgrid_test PRIVATE
  "${PROJECT_SOURCE_DIR}/test"
  "${PROJECT_SOURCE_DIR}/third_party")
target_link_libraries(muda_spgrid_test PRIVATE muda Eigen3::Eigen)

add_test(NAME muda_spgrid_test COMMAND muda_spgrid_test)
