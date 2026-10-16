# The testxgboost executable is created in the top level CMakeLists. Most of the
# properties and compilation flags are already set. We just need to add source files and
# link gtest here.

# find/build GTest
include(${xgboost_SOURCE_DIR}/cmake/CPMFindGTest.cmake)

file(GLOB_RECURSE TEST_SOURCES "*.cc")

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_TEST_SOURCES "*.cu")
  list(APPEND TEST_SOURCES ${CUDA_TEST_SOURCES})
endif()

file(GLOB_RECURSE SYCL_TEST_SOURCES "plugin/test_sycl_*.cc")
list(REMOVE_ITEM TEST_SOURCES ${SYCL_TEST_SOURCES})

if(PLUGIN_SYCL)
  set(CMAKE_CXX_COMPILER "${SYCL_CXX_COMPILER_PATH}")
  file(GLOB_RECURSE SYCL_TEST_SOURCES "plugin/test_sycl_*.cc")
  add_library(plugin_sycl_test OBJECT ${SYCL_TEST_SOURCES})

  target_include_directories(plugin_sycl_test
    PRIVATE
    ${gtest_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/dmlc-core/include)

  target_compile_definitions(plugin_sycl_test PUBLIC -DXGBOOST_USE_SYCL=1)
  target_link_libraries(plugin_sycl_test PRIVATE GTest::gtest GTest::gmock)
  xgboost_sycl_object_library(plugin_sycl_test)
  # Get compilation flags of plugin_sycl_test and propagate to testxgboost
  target_link_libraries(testxgboost PUBLIC plugin_sycl_test plugin_sycl_test_device_link)
  # Add all objects of plugin_sycl to testxgboost
  target_sources(testxgboost INTERFACE $<TARGET_OBJECTS:plugin_sycl_test>)
endif()

target_sources(
  testxgboost PRIVATE
  ${TEST_SOURCES}
  ${xgboost_SOURCE_DIR}/plugin/example/custom_obj.cc
)

if(USE_CUDA AND PLUGIN_RMM)
  target_include_directories(testxgboost PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

target_include_directories(testxgboost
  PRIVATE
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include)
target_link_libraries(testxgboost
  PRIVATE
  $<TARGET_NAME_IF_EXISTS:rmm::rmm_logger>
  $<TARGET_NAME_IF_EXISTS:rmm::rmm_logger_impl>
  GTest::gtest GTest::gmock)

set_output_directory(testxgboost ${xgboost_BINARY_DIR})

# This grouping organises source files nicely in visual studio
auto_source_group("${TEST_SOURCES}")
