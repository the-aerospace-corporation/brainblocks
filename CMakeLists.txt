cmake_minimum_required(VERSION 3.1)

project(BrainBlocks VERSION 0.3.7)


# BB Core Source Files, all standard C
set(SOURCE_FILES
    src/bbcore/utils.c
    src/bbcore/bitarray.c
    src/bbcore/page.c
    src/bbcore/coincidence_set.c
    src/bbcore/blank_block.c
    src/bbcore/scalar_encoder.c
    src/bbcore/symbols_encoder.c
    src/bbcore/persistence_encoder.c
    src/bbcore/pattern_classifier.c
    src/bbcore/pattern_pooler.c
    src/bbcore/sequence_learner.c
)

# BB Core Header Files
include_directories(src/bbcore/ src/)

# If called to build a python extension with pybind11
if (PYTHON_EXTENSION)

    # Build the python extension module
    # --------------------------------

    # bbcore source files, all standard C
    add_library(bbcore ${SOURCE_FILES})

    # pybind11 and cmake stuff
    add_subdirectory(src/3rdparty/pybind11)

    # pybind11 wrapper and C++ code
    pybind11_add_module(bb_backend src/wrappers/python_bindings.cpp)

    # set_property(TARGET bb_backend PROPERTY CXX_STANDARD 17)

    # link bbcore to C++/Python interface
    target_link_libraries(bb_backend PRIVATE bbcore)

# otherwise, build the native binaries
else()

    # David's custom environment changes preserved
    # TODO: these from some template or actual requirements that we need?
    # --------------------------------------

    # set output directory as bin/
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
    #set(CMAKE_C_STANDARD 11)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_SHARED_LIBRARY_PREFIX "")

    # set optimization flag
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_C_FLAGS "/O2")
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    else()
        set(CMAKE_C_FLAGS "-O3")
    endif()

    if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release)
    endif()

    # Build the executables
    # ------------------------

    # bbcore source files, all standard C
    add_library(bbcore ${SOURCE_FILES})

    # create C binary target
    add_executable(brainblocks tests/c/test.c)

    # link bbcore to C binary
    target_link_libraries(brainblocks PRIVATE bbcore)

    # create C++ binary target
    add_executable(brainblocks_cpp tests/cpp/test.cpp)

    # link bbcore to C++ binary
    target_link_libraries(brainblocks_cpp PRIVATE bbcore)

endif()
