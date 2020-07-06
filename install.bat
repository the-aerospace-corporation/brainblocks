@echo off

REM clear previous build
rmdir /s /q bin > nul 2>&1
rmdir /s /q b build > nul 2>&1
rmdir /s /q b dist > nul 2>&1
mkdir bin
mkdir build
mkdir dist

REM change to build directory
cd build

echo ================================================================================
echo Compiling BrainBlocks Backend
echo ================================================================================

cmake ..
cmake --build . --config Release

echo .
echo ================================================================================
echo Installing BrainBlocks Python Bindings
echo ================================================================================

cd ..

REM install python requirements
pip install -r requirements.txt

REM uninstall brainblocks from environment
pip uninstall brainblocks -y

REM create wheel package and install (OPTION #1)
python setup.py bdist_wheel
rmdir /s /q "brainblocks.egg-info"
FOR %%I in (dist\*.whl) DO pip install %%I 


REM pip build and install the directory (OPTION #2)
REM pip install .

REM build egg package and install (OPTION #3)
REM python setup.py build
REM python setup.py install
