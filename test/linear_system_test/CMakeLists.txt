# linear system test
find_package(Eigen3 REQUIRED)
file(GLOB_RECURSE SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
add_executable(muda_linear_system_test)
target_sources(muda_linear_system_test PRIVATE ${SOURCE} ${MUDA_HEADER_FILES})
target_compile_features(muda_linear_system_test PRIVATE cxx_std_20)
set_target_properties(muda_linear_system_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(muda_linear_system_test PRIVATE
  "${PROJECT_SOURCE_DIR}/test"
  "${PROJECT_SOURCE_DIR}/third_party")
target_link_libraries(muda_linear_system_test PRIVATE muda cusparse cublas cusolver Eigen3::Eigen)
source_group(TREE "${PROJECT_SOURCE_DIR}/test" PREFIX "test" FILES ${SOURCE})
source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "src" FILES ${MUDA_HEADER_FILES})

add_test(NAME muda_linear_system_test COMMAND muda_linear_system_test)
