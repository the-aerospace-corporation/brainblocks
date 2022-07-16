# ==============================================================================
# install.py
# ==============================================================================
import argparse
import os
import platform
import shutil
import subprocess
import struct
import sys
from pathlib import Path


def is_python_64bit():
    return (struct.calcsize('P') == 8)


def rm_r(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
    else:
        shutil.rmtree(path)

# ==============================================================================
# Install
#
# Installs Python BrainBlocks to the environment
# ==============================================================================
def install():

    # Uninstall Python BrainBlocks if it already exists
    uninstall()

    # Clean
    clean()

    # Create wheel package
    build()

    # Install Python BrainBlocks
    print('=' * 80)
    print('Install Wheel Package')
    print('=' * 80, flush=True)
    shutil.rmtree('brainblocks.egg-info', ignore_errors=True)
    wheel_path = next(Path("dist").glob("*.whl"))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", str(wheel_path)])

    # Clean
    clean()

# ==============================================================================
# Build
#
# Builds Python wheel package
# ==============================================================================
def build():
    # Create wheel package
    print('=' * 80)
    print('Build Python Packages')
    print('=' * 80, flush=True)
    subprocess.check_call([sys.executable, "-m", "build"])

# ==============================================================================
# Uninstall
#
# Uninstalls Python BrainBlocks from environment
# ==============================================================================
def uninstall():

    # Uninstall Python BrainBlocks if it already exists
    print('=' * 80)
    print('Uninstall Any Existing BrainBlocks')
    print('=' * 80, flush=True)
    result = subprocess.check_output([sys.executable, "-m", "pip", "list"])
    if "brainblocks" in str(result):
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'uninstall', 'brainblocks', '-y'])


# ==============================================================================
# Build C++ Tests
#
# Compiles C++ BrainBlocks tests
# ==============================================================================
def cpptests():

    # Clear previous build
    for directory in ['build', 'dist']:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)

    # Navigate to build directory
    os.chdir('build')

    # Get system name
    uname_obj = platform.uname()

    # Get system type
    if is_python_64bit():
        ARCH='x64'
    else:
        ARCH='x32'

    # Generating BrainBlocks build system
    print('=' * 80)
    print('Generating BrainBlocks build system')
    print('=' * 80, flush=True)
    if uname_obj.system == 'Windows':
        cmd = ['cmake', '-DCMAKE_GENERATOR_PLATFORM=%s' % ARCH,
               '-DBRAINBLOCKS_TESTS=true', '..']
    else:
        cmd = ['cmake', '-DBRAINBLOCKS_TESTS=true', '..']

    if subprocess.call(cmd) != 0:
        print('ERROR while cmake configure')
        sys.exit(-1)

    # Building BrainBlocks
    print('=' * 80)
    print('Building BrainBlocks')
    print('=' * 80, flush=True)
    cmd = ['cmake', '--build', '.', '--config', 'Release']

    if subprocess.call(cmd) != 0:
        print('ERROR while cmake build')
        sys.exit(-1)

    os.chdir('..')

# ==============================================================================
# Clean
#
# Remove all known artifacts from CMake and setuptools build processes
# ==============================================================================
def clean():

    print('=' * 80)
    print('Clean Previous Builds')
    print('=' * 80, flush=True)

    directories = [
        'cmake_install.cmake',
        'brainblocks.egg-info',
        'CMakeCache.txt',
        'Makefile',
        'CMakeFiles',
        'bin',
        'build',
        'dist',
        '.pytest_cache']

    for directory in directories:
        rm_r(directory)

    for wildcard in ['*.a', '*.so', '*.lib', '*.dll', '*.egg-info']:
        for path in Path('.').rglob(wildcard):
            rm_r(path.name)

# ==============================================================================
# Tests
#
# Run Python unit tests
# ==============================================================================
def test():

    print('=' * 80)
    print('Run Python Tests')
    print('=' * 80, flush=True)
    subprocess.check_call([sys.executable, "-m", "pytest"])

# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':

    # Handle argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--install', action='store_true',
                        help='Installs Python BrainBlocks')

    parser.add_argument('--uninstall', action='store_true',
                        help='Uninstalls Python BrainBlocks')

    parser.add_argument('--build', action='store_true',
                        help='Build Python Packages')

    parser.add_argument('--clean', action='store_true',
                        help='Cleans up project directory')

    parser.add_argument('--test', action='store_true',
                        help='Run Python Unit Tests')

    parser.add_argument('--cpptests', action='store_true',
                        help='Compiles C++ unit tests')

    args = parser.parse_args()

    # Handle functions
    if args.install:
        install()

    if args.uninstall:
        uninstall()

    if args.build:
        build()

    if args.clean:
        clean()

    if args.test:
        test()

    if args.cpptests:
        cpptests()

