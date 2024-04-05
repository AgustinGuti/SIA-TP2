import os
import shutil
import subprocess
import stat

# Install the requirements
subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

# Move the Git hooks
git_hooks_dir = os.path.join(".git", "hooks")
source_hooks_dir = "./git_hooks"

for filename in os.listdir(source_hooks_dir):
    source = os.path.join(source_hooks_dir, filename)
    destination = os.path.join(git_hooks_dir, filename)
    shutil.copy(source, destination)
    os.chmod(destination, os.stat(destination).st_mode | stat.S_IEXEC)