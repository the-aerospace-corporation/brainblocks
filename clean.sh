#!/usr/bin/env bash
# Remove all known artifacts from CMake and setuptools build processes
rm -f cmake_install.cmake
rm -f CMakeCache.txt
rm -f Makefile
rm -f *.a
rm -f *.so
rm -f *.lib
rm -f *.dll
rm -rf CMakeFiles
rm -rf bin
rm -rf build
rm -rf dist
rm -rf *.egg-info
rm -rf .pytest_cache
