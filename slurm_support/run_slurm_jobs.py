import os
import subprocess
import sys

def run_jobs(direc: str):
    os.chdir(direc)
    fnames = os.listdir()
    for fn in fnames:
        if not fn.endswith('.txt'): continue
        print(subprocess.run(['sbatch', fn], capture_output=True))

if __name__ == '__main__':
    run_jobs(sys.argv[1])
