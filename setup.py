# ==============================================================================
# setup.py
# ==============================================================================
import os
import re
import sys
import platform
import sysconfig
import subprocess
import pprint

from setuptools import setup, Extension, find_packages, find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from distutils.version import LooseVersion


# ==============================================================================
# PyTest
# ==============================================================================
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


# ==============================================================================
# CMakeExtension
# ==============================================================================
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


# ==============================================================================
# CMakeBuild
# ==============================================================================
class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        # Remove old cache files if they exist
        try:
            os.remove("CMakeCache.txt")
        except:
            pass

        # Build extensions
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        cmake_args = []
        build_args = []
        cxxflags = ''

        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        cfg = 'Debug' if self.debug else 'Release'

        # Set platform-independent arguments
        cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir]
        cmake_args += ['-DPYTHON_EXTENSION=1']
        cmake_args += ['-DPYTHON_EXECUTABLE=' + sys.executable]
        cmake_args += ['-DPYTHON_PATHS=' + ';'.join(sys.path)]
        build_args += ['--config', cfg]
        cxxflags += ' -DVERSION_INFO=\\"{}\\"'.format(
            self.distribution.get_version())

        env = os.environ.copy()

        # If Windows
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(), extdir)]
            # If 64-bit then specifiy the platform name
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
            cxxflags += ' /O2'

        # If MacOS
        elif platform.system() == "Darwin":
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']
            cxxflags += ' -O3 -g -fPIC'
            # cxxflags += ' -Wall -Wextra' # for build warnings

            # macos deployment target same as python ABI version
            config_vars = sysconfig.get_config_vars()
            env['MACOSX_DEPLOYMENT_TARGET'] = config_vars['MACOSX_DEPLOYMENT_TARGET']

        # assume Linux
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']
            cxxflags += ' -O3 -g -fPIC'
            # cxxflags += ' -Wall -Wextra' # for build warnings

        # C++ flags
        env['CXXFLAGS'] = cxxflags

        # Make current working directory if it doesnt already exist
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Generating C++ BrainBlocks build system
        print('=' * 80)
        print('Generating C++ BrainBlocks build system')
        print('=' * 80, flush=True)
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)

        # Building C++ BrainBlocks
        print('=' * 80)
        print('Building C++ BrainBlocks')
        print('=' * 80, flush=True)
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)


# ==============================================================================
# Setup
# ==============================================================================

# Create long description from README
with open("README.md", "r") as fh:
    long_description = fh.read()

# Setup
setup(
    cmdclass=dict(build_ext=CMakeBuild, test=PyTest),
    ext_modules=[CMakeExtension('brainblocks.bb_backend')]
)
