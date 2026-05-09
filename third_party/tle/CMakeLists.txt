
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(External)

# flagtree tle
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include)

get_or_compile_flagcx()

add_subdirectory(dialect)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonTLE
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle_raw.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/lib/Protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/lib/AnalyzeReturnType.cpp
    LINK_LIBS
    MLIRTargetLLVMIRImport
    TritonTLETransforms
    TritonIR
    )
  if(USE_FLAGCX)
    add_dependencies(TritonTLE flagcx)
    target_link_libraries(TritonTLE PRIVATE Python3::Module pybind11::headers MLIRTargetLLVMIRImport PUBLIC flagcx)
    target_link_options(TritonTLE PUBLIC -Wl,--no-as-needed)
  else()
    target_link_libraries(TritonTLE PRIVATE Python3::Module pybind11::headers MLIRTargetLLVMIRImport)
  endif()
endif()

add_subdirectory(test)
