import shutil
import os
from pathlib import Path


def rm_r(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
    else:
        shutil.rmtree(path)

# Remove all known artifacts from CMake and setuptools build processes
for directory in ['cmake_install.cmake', 'CMakeCache.txt', 'Makefile', 'CMakeFiles', 'bin', 'build', 'dist', '.pytest_cache']:
    rm_r(directory)

for wildcard in ['*.a', '*.so', '*.lib', '*.dll', '*.egg-info']:
    for path in Path('.').rglob(wildcard):
        rm_r(path.name)
