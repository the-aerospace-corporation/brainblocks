@echo off

REM Remove all known artifacts from CMake and setuptools build processes
rmdir /s /q cmake_install.cmake > nul 2>&1
rmdir /s /q CMakeCache.txt > nul 2>&1
rmdir /s /q Makefile > nul 2>&1
rmdir /s /q *.a > nul 2>&1
rmdir /s /q *.so > nul 2>&1
rmdir /s /q *.lib > nul 2>&1
rmdir /s /q *.dll > nul 2>&1
rmdir /s /q CMakeFiles > nul 2>&1
rmdir /s /q bin > nul 2>&1
rmdir /s /q build > nul 2>&1
rmdir /s /q dist > nul 2>&1
rmdir /s /q *.egg-info > nul 2>&1
rmdir /s /q .pytest_cache > nul 2>&1
