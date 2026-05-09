function(_set_flagcx_include_directories)
      set(FLAGCX_INCLUDE_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flagcx/flagcx")
      include_directories(${FLAGCX_INCLUDE_PATH}/include)
      include_directories(${FLAGCX_INCLUDE_PATH}/adaptor/include)
      include_directories(${FLAGCX_INCLUDE_PATH}/service/include)
      include_directories(${FLAGCX_INCLUDE_PATH}/core/include)
endfunction()

function(get_flagcx_lib_from_cache)
  set(FLAGCX_CACHE_PATH "$ENV{HOME}/.flagtree/flagcx/libflagcx.so" CACHE PATH "Path to flagcx")
  find_library(FLAGCX_LIB
    NAMES flagcx
    PATHS ${FLAGCX_CACHE_PATH}
  )
endfunction()


function(copy_flagcx_lib_to_cache SRC_PATH MODE)
    if(MODE STREQUAL "build")
        add_custom_command(TARGET flagcx_ext POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${SRC_PATH}
                    ${FLAGCX_CACHE_PATH}
        )
    elseif(MODE STREQUAL "common")
        file(COPY ${SRC_PATH} DESTINATION ${FLAGCX_CACHE_PATH})
    else()
        message(FATAL_ERROR "Unknown MODE: ${MODE}")
    endif()
endfunction()

function(_get_flagcx)
  set(FLAGCX_LIB_PATH "$ENV{FLAGCX_LIB_PATH}")
  if(EXISTS ${FLAGCX_LIB_PATH})
    message(STATUS "Using precompiled FlagCX library at ${FLAGCX_LIB_PATH}")
    set_target_properties(flagcx PROPERTIES IMPORTED_LOCATION ${FLAGCX_LIB_PATH})
    set(FLAGCX_LIB "${FLAGCX_LIB_PATH}/libflagcx.so" cache STRING "")
    copy_flagcx_lib_to_cache(${FLAGCX_LIB_PATH} "common")
  else()
    get_flagcx_lib_from_cache()
    if(${FLAGCX_LIB})
      message(STATUS "Using cached FlagCX library at ${FLAGCX_LIB}")
    else()
    message(STATUS "Specified FLAGCX_LIB_PATH does not exist: ${FLAGCX_LIB_PATH},
      please check the path and try again or unset the environment variable 'FLAGCX_LIB_PATH'
      to compile FlagCX from source.")
    endif()
  endif()
endfunction()

function(_compile_flagcx)
    set(FLAGCX_PROJECT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flagcx")
    if(EXISTS ${FLAGCX_PROJECT_PATH} AND IS_DIRECTORY ${FLAGCX_PROJECT_PATH})
      message(STATUS "Enable the tle-distribute capability using FlagCX")
      _set_flagcx_include_directories()
      set(LIBFLAGCX ${FLAGCX_PROJECT_PATH}/build/lib/libflagcx.so)
      include(ExternalProject)
      ExternalProject_Add(
        flagcx_ext
        SOURCE_DIR ${FLAGCX_PROJECT_PATH}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND make
        INSTALL_COMMAND ""
        BUILD_IN_SOURCE 1
        BUILD_BYPRODUCTS ${LIBFLAGCX}
      )
      add_library(flagcx SHARED IMPORTED GLOBAL)
      add_dependencies(flagcx flagcx_ext)
      set_target_properties(flagcx PROPERTIES
        IMPORTED_LOCATION ${LIBFLAGCX}
      )
      copy_flagcx_lib_to_cache(${LIBFLAGCX} "build")
    else()
      message(FATAL_ERROR "Required directory 'flagcx' not found at ${FLAGCX_PROJECT_PATH}")
    endif()

endfunction()

function(get_or_compile_flagcx)
  set(USE_FLAGCX "$ENV{USE_FLAGCX}" cache STRING "")
  if(USE_FLAGCX)
    _get_flagcx()
    if (NOT FLAGCX_LIB)
      _compile_flagcx()
    endif()
    _set_flagcx_include_directories()
  endif()
endfunction()
