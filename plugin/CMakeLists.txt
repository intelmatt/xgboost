if(PLUGIN_SYCL)
  set(CMAKE_CXX_COMPILER "${SYCL_CXX_COMPILER_PATH}")
  file(GLOB_RECURSE SYCL_SOURCES "sycl/*.cc")
  add_library(plugin_sycl OBJECT ${SYCL_SOURCES})
  target_include_directories(plugin_sycl
    PRIVATE
    ${xgboost_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/dmlc-core/include
    ${xgboost_SOURCE_DIR}/rabit/include)
    target_compile_definitions(plugin_sycl PUBLIC -DXGBOOST_USE_SYCL=1)
    target_compile_options(plugin_sycl PRIVATE -fno-sycl-id-queries-fit-in-int)
  xgboost_sycl_object_library(plugin_sycl)
  # Get compilation flags of plugin_sycl and propagate to objxgboost
  target_link_libraries(objxgboost PUBLIC plugin_sycl plugin_sycl_device_link)
  # Add all objects of plugin_sycl to objxgboost
  target_sources(objxgboost INTERFACE $<TARGET_OBJECTS:plugin_sycl>)
endif()
