# ==============================================================================
# CMakeLists.txt
# ==============================================================================
cmake_minimum_required(VERSION 3.7)
project(BRAINBLOCKS)

# Default flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable fopen secure warnings
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Build bbcore
add_subdirectory(src/cpp)

# If called then build C++ tests
if (BRAINBLOCKS_TESTS)
    add_subdirectory(tests/cpp)
endif()

# If called then build a python extension with pybind11
if(PYTHON_EXTENSION)

    include_directories(${BRAINBLOCKS_SOURCE_DIR}/src/cpp)

    # Cmake pybind11
    add_subdirectory(src/3rdparty/pybind11)

    # Handle python wrapper
    pybind11_add_module(bb_backend src/wrappers/python_bindings.cpp)

    # Link bbcore to C++/Python interface
    target_link_libraries(bb_backend PRIVATE bbcore)

endif()
