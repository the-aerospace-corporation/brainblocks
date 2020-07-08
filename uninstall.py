import subprocess
import sys

# uninstall brainblocks from environment
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "brainblocks", "-y"])