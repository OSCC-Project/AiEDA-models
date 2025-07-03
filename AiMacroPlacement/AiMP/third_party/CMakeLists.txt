cmake_minimum_required(VERSION 3.0)

set(CMAKE_VERBOSE_MAKEFILE ON)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Prefix prepended to install directories" FORCE )
endif()
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# This is a dirty fix for CMake on some machines that generates default CXX standard and override the custom settings,
# because CUDA does not support C++17 and higher.
# You can check the CMAKE_CXX_FLAGS in CMakeCache.txt to verify the issue.
if(CMAKE_CXX_FLAGS)
string(REGEX REPLACE "-std=c\\+\\+([0-9a-z]+)" " " CMAKE_CXX_FLAGS_INIT ${CMAKE_CXX_FLAGS})
endif(CMAKE_CXX_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_INIT}" CACHE STRING "Flags used by the compiler during all build types." FORCE)

set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG") # 
if(NOT CMAKE_CXX_ABI)
    set(CMAKE_CXX_ABI 0 CACHE STRING
        "Choose the value for _GLIBCXX_USE_CXX11_ABI, options are: 0|1."
        FORCE)
endif(NOT CMAKE_CXX_ABI)
message(STATUS "CMAKE_CXX_ABI: _GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI})


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# without this, clang will complain about linking
#set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE TRUE)

set(IEDA_BUILD_AS_THIRDPARTY ON)
add_subdirectory(iEDA)

# add_subdirectory(AutoDMP)


# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dreamplace/configure.py DESTINATION dreamplace)
