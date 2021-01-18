# ==============================================================================
# setup.py
# ==============================================================================
import os
import re
import sys
import platform
import subprocess

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
        build_args += ['--config', cfg]
        cxxflags += ' -DVERSION_INFO=\\"{}\\"'.format(
                    self.distribution.get_version())

        # If Windows
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                          cfg.upper(), extdir)]
            # If 64-bit then specifiy the platform name
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
            cxxflags += ' /O2'

        # If Linux or MacOS
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']
            cxxflags += ' -O3 -g -fPIC'
            #cxxflags += ' -Wall -Wextra' # for build warnings

        # Setup environment
        env = os.environ.copy()
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
            ['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# ==============================================================================
# Setup
# ==============================================================================

# Create long description from README
with open("README.md", "r") as fh:
    long_description = fh.read()

# Setup
setup(
    name="brainblocks",
    version="0.7.0",
    packages=[
        "brainblocks",
        "brainblocks.blocks",
        "brainblocks.datasets",
        "brainblocks.metrics",
        "brainblocks.templates",
        "brainblocks.tools",
    ],
    package_dir={'brainblocks': 'src/python'},
    python_requires=">=3.6",
    author="Jacob Everist, David Di Giorgio",
    author_email="jacob.s.everist@aero.org, david.digiorgio@aero.org",
    description="BrainBlocks Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://aerosource2.aero.org/bitbucket/projects/BRAINBLOCKS/repos/brainblocks-c",
    keywords="brainblocks htm classification anomaly abnormality time-series neuroscience cognitive distributed representation",

    project_urls={
        "JIRA": "https://aerosource2.aero.org/jira/projects/BRAINBLOCK",
        "Confluence": "https://aerosource2.aero.org/confluence/display/BRAINBLOCKS/BrainBlocks",
        "Source Code": "https://aerosource2.aero.org/bitbucket/projects/BRAINBLOCKS/repos/brainblocks",
    },
    install_requires=[
        #'numpy',
        #'scipy',
        #'sklearn'
    ],
    tests_require=['pytest'],
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha"
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    cmdclass=dict(build_ext=CMakeBuild, test=PyTest),
    # cmdclass=dict(build_ext=CMakeBuild),
    ext_modules=[CMakeExtension('brainblocks.bb_backend')],
    zip_safe=False,
)
