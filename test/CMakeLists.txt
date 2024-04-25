# common unit test
find_package(Eigen3 REQUIRED)
file(GLOB_RECURSE MUDA_UNIT_TEST_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/test/unit_test/*.cpp"
  "${PROJECT_SOURCE_DIR}/test/unit_test/*.cu"
  "${PROJECT_SOURCE_DIR}/test/unit_test/*.h")
add_executable(muda_unit_test ${MUDA_UNIT_TEST_SOURCE_FILES})
set_target_properties(muda_unit_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(muda_unit_test PRIVATE
  "${PROJECT_SOURCE_DIR}/test"
  "${PROJECT_SOURCE_DIR}/external")
target_link_libraries(muda_unit_test PRIVATE muda cusparse cublas cusolver Eigen3::Eigen)
target_compile_definitions(muda_unit_test PRIVATE "-DMUDA_TEST_DATA_DIR=R\"(${PROJECT_SOURCE_DIR}/test/data)\"")
source_group(TREE "${PROJECT_SOURCE_DIR}/test" PREFIX "test" FILES ${MUDA_UNIT_TEST_SOURCE_FILES})
source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "src" FILES ${MUDA_HEADER_FILES})

# check eigen validation in cuda
find_package(Eigen3 REQUIRED)
file(GLOB_RECURSE MUDA_EIGEN_TEST_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/test/eigen_test/*.cpp"
  "${PROJECT_SOURCE_DIR}/test/eigen_test/*.cu"
  "${PROJECT_SOURCE_DIR}/test/eigen_test/*.h")
add_executable(muda_eigen_test ${MUDA_EIGEN_TEST_SOURCE_FILES})
set_target_properties(muda_eigen_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(muda_eigen_test PRIVATE
  "${PROJECT_SOURCE_DIR}/test"
  "${PROJECT_SOURCE_DIR}/external")
target_link_libraries(muda_eigen_test PRIVATE muda Eigen3::Eigen)
source_group(TREE "${PROJECT_SOURCE_DIR}/test" PREFIX "test" FILES ${MUDA_EIGEN_TEST_SOURCE_FILES})
source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "src" FILES ${MUDA_HEADER_FILES})

# linear system test
find_package(Eigen3 REQUIRED)
file(GLOB_RECURSE MUDA_LINEAR_SYSTEM_TEST_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/test/linear_system_test/*.cpp"
  "${PROJECT_SOURCE_DIR}/test/linear_system_test/*.cu"
  "${PROJECT_SOURCE_DIR}/test/linear_system_test/*.h")
add_executable(muda_linear_sysytem_test ${MUDA_LINEAR_SYSTEM_TEST_SOURCE_FILES})
set_target_properties(muda_linear_sysytem_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(muda_linear_sysytem_test PRIVATE
  "${PROJECT_SOURCE_DIR}/test"
  "${PROJECT_SOURCE_DIR}/external")
target_link_libraries(muda_linear_sysytem_test PRIVATE muda cusparse cublas cusolver Eigen3::Eigen)
source_group(TREE "${PROJECT_SOURCE_DIR}/test" PREFIX "test" FILES ${MUDA_LINEAR_SYSTEM_TEST_SOURCE_FILES})
source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "src" FILES ${MUDA_HEADER_FILES})

if(MSVC)
  # when using c++ compiler, ignore C4819
  set(disable_warning -Xcompiler "/wd 4819")
  target_compile_options(muda_unit_test PRIVATE ${disable_warning})
  target_compile_options(muda_eigen_test PRIVATE ${disable_warning})
  target_compile_options(muda_linear_sysytem_test PRIVATE ${disable_warning})
endif()