#!/usr/bin/env bash

# clear previous build
rm -rf bin; mkdir bin
rm -rf build; mkdir build
rm -rf dist; mkdir dist

# change to build directory
cd build

printf "================================================================================\n"
printf "Compiling BrainBlocks Backend\n"
printf "================================================================================\n"

printf "OSTYPE=%s\n" ${OSTYPE}
if [ ${OSTYPE} == 'msys' ]; then
    ARCH="x32"
    MTYPE=`uname -m`
    printf "MTYPE=%s\n" ${MTYPE}
    if [ ${MTYPE} == 'x86_64' ]; then
        ARCH="x64"
    fi
    cmake -DCMAKE_GENERATOR_PLATFORM=${ARCH} ..
else
    cmake ..
fi

cmake --build . --config Release

printf "\n"
printf "================================================================================\n"
printf "Installing BrainBlocks Python Bindings\n"
printf "================================================================================\n"

cd ..

# install python requirements
pip install -r requirements.txt

# uninstall brainblocks from environment
pip uninstall brainblocks -y

# create wheel package and install (OPTION #1)
python setup.py bdist_wheel
rm -r brainblocks.egg-info/
pip install dist/*.whl

# pip build and install the directory (OPTION #2)
#pip install .

# build egg package and install (OPTION #3)
#python setup.py build
#python setup.py install
