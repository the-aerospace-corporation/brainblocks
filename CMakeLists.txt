cmake_minimum_required(VERSION 3.1)

project(BrainBlocks VERSION 0.3.7)

# BB Core Source Files
set(SOURCE_FILES
    src/bbcore/utils.cpp
    src/bbcore/bitarray.cpp
    src/bbcore/page.cpp
    src/bbcore/coincidence_set.cpp
    src/bbcore/blank_block.cpp
    src/bbcore/scalar_encoder.cpp
    src/bbcore/symbols_encoder.cpp
    src/bbcore/persistence_encoder.cpp
    src/bbcore/pattern_classifier.cpp
    src/bbcore/pattern_classifier_dynamic.cpp
    src/bbcore/pattern_pooler.cpp
    src/bbcore/sequence_learner.cpp
    src/bbcore/context_learner.cpp
)

# BB Core Header Files
include_directories(src/bbcore/ src/)

# bbcore C++ source files
add_library(bbcore ${SOURCE_FILES})

# if called to build a python extension with pybind11
if (PYTHON_EXTENSION)

    # pybind11 and cmake stuff
    add_subdirectory(src/3rdparty/pybind11)

    # pybind11 wrapper and C++ code
    pybind11_add_module(bb_backend src/wrappers/python_bindings.cpp)

    # set_property(TARGET bb_backend PROPERTY CXX_STANDARD 17)

    # link bbcore to C++/Python interface
    target_link_libraries(bb_backend PRIVATE bbcore)

# otherwise, build the native binaries
else()

    # set bin/ as output directory
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BrainBlocks_SOURCE_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BrainBlocks_SOURCE_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BrainBlocks_SOURCE_DIR}/bin)
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${BrainBlocks_SOURCE_DIR}/bin)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${BrainBlocks_SOURCE_DIR}/bin)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${BrainBlocks_SOURCE_DIR}/bin)
    endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

    # set compiler options
    ##set(CMAKE_C_STANDARD 11)
    set(CMAKE_CXX_STANDARD 11)
    #set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    #set(CMAKE_SHARED_LIBRARY_PREFIX "")

    # set optimization flag
    #if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    #    set(CMAKE_C_FLAGS "/O2")
    #    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    #else()
    #    set(CMAKE_C_FLAGS "-O3")
    #endif()

    #if(NOT CMAKE_BUILD_TYPE)
    #  set(CMAKE_BUILD_TYPE Release)
    #endif()

    # create C++ binary target
    add_executable(brainblocks_cpp tests/cpp/test.cpp)

    # link bbcore to C++ binary
    target_link_libraries(brainblocks_cpp PRIVATE bbcore)

endif()
