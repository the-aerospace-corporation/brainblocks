import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages, find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from distutils.version import LooseVersion


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


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        # remove old cache files if they exist, screws up build if they're there from alternative build
        try:
            os.remove("CMakeCache.txt")
        except:
            pass

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXTENSION=1',
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -O0 -g -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                                     self.distribution.get_version())
        env['CFLAGS'] = '-O0 -g -fPIC'

        # build_args += ['-O0', '-g']
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


# CFLAGS="-O0 -g" CXXFLAGS="-O0 -g" python setup.py install

# create long description from README
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="brainblocks",
    version="0.3.7",
    packages=["brainblocks", "brainblocks.tools", "brainblocks.datasets", "brainblocks.metrics", "brainblocks.blocks",
              "brainblocks.templates"],
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
        "Source Code": "https://aerosource2.aero.org/bitbucket/projects/BRAINBLOCKS/repos/brainblocks-c",
    },
    install_requires=['numpy', 'scipy', 'sklearn'],
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
