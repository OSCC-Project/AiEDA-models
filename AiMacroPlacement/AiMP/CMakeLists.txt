# CMake Version Requirements
# SET (CMAKE_C_COMPILER_WORKS 1)

# SET (CMAKE_CXX_COMPILER_WORKS 1)
cmake_minimum_required(VERSION 3.16)
project(iEDA3D-placer)

set(AI_EDA_TOP_DIR ${PROJECT_SOURCE_DIR})
# add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_CXX_STANDARD 20)
set(PYTHON_EXECUTABLE "/home/zhaoxueyan/anaconda3/envs/iEDA-DSE/bin/python")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_PREFIX ${AI_EDA_TOP_DIR}/third_party/iEDA/bin)

# Preset Compilation Conditions
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

set(CMAKE_CXX_ABI "1")

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG") # 
set(CMAKE_VERBOSE_MAKEFILEON ON)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb ") # -Wl --start-group

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party)

# Set
include_directories(${PROJECT_SOURCE_DIR})
