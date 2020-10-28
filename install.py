import os
import platform
import shutil
import subprocess
import struct
import sys
from pathlib import Path

# clear previous build
for directory in ['bin', 'build', 'dist']:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

# change to build directory
os.chdir('build')

print("================================================================================")
print("Compiling BrainBlocks Backend")
print("================================================================================")

uname_obj = platform.uname()
print(uname_obj.system)

def is_python_64bit():
    return (struct.calcsize("P") == 8)

if is_python_64bit():
    ARCH="x64"
else:
    ARCH="x32"

if uname_obj.system == 'Windows':
    cmd = ["cmake", '-DCMAKE_GENERATOR_PLATFORM=%s' % ARCH, ".."]
else:
    cmd = ["cmake", ".."]
if subprocess.call(cmd) != 0:
    print("ERROR while cmake configure")
    sys.exit(-1)

cmd = ["cmake", '--build', ".", '--config', 'Release']
if subprocess.call(cmd) != 0:
    print("ERROR while cmake build")
    sys.exit(-1)

'''
print("")
print("================================================================================")
print("Installing BrainBlocks Python Bindings")
print("================================================================================")

os.chdir('..')

# install python requirements
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# uninstall brainblocks from environment if it exists
result = subprocess.check_output([sys.executable, "-m", "pip", "list"])
if "brainblocks" in str(result):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "brainblocks", "-y"])

# create wheel package and install
subprocess.check_call([sys.executable, "setup.py", "bdist_wheel"])

shutil.rmtree('brainblocks.egg-info')

wheel_path = next(Path("dist").glob("*.whl"))

subprocess.check_call([sys.executable, "-m", "pip", "install", str(wheel_path)])
'''