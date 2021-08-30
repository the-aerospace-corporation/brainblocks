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

    set(Python_FIND_REGISTRY NEVER)
    #set(CMAKE_FIND_FRAMEWORK NEVER)
    set(Python_FIND_FRAMEWORK NEVER)

    #message(STATUS "print_all_variables------------------------------------------{")
    #get_cmake_property(_variableNames VARIABLES)
    #foreach (_variableName ${_variableNames})
    #    message(STATUS "${_variableName}=${${_variableName}}")
    #endforeach()
    #message(STATUS "print_all_variables------------------------------------------}")

    # find pybind11 in python libraries
    find_package(pybind11 REQUIRED PATHS ${PYTHON_PATHS})

    include_directories(${BRAINBLOCKS_SOURCE_DIR}/src/cpp)

    # Handle python wrapper
    pybind11_add_module(bb_backend src/wrappers/python_bindings.cpp)

    # Link bbcore to C++/Python interface
    target_link_libraries(bb_backend PRIVATE bbcore)

endif()
