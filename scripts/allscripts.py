import subprocess
import shlex

subprocess.run(shlex.split("python3 scripts/batchscript.py"))
subprocess.run(shlex.split("python3 scripts/lrscript.py"))
subprocess.run(shlex.split("python3 scripts/momentumscript.py"))
subprocess.run(shlex.split("python3 optimizerscript.py"))
subprocess.run(shlex.split("python3 transizescript.py"))
subprocess.run(shlex.split("python3 weightdecayscript.py"))