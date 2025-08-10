#!/bin/bash
#SBATCH -J jupyter
#SBATCH -N 1
#SBATCH --mem=10GB
#SBATCH --cpus-per-task 20
#SBATCH --time=3-1:10:10
#SBATCH -e /project/Stat/s1155168529/programs/DDML/output/test_jupyter_error40nogpu.log
#SBATCH --output=/project/Stat/s1155168529/programs/DDML/output/jupyter40nogpu.log

cd /users/s1155168529/
source ~/.bashrc
conda activate py_ddml

cat /etc/hosts

VSCODE_IPC_HOOK_CLI=$( lsof | grep $UID/vscode-ipc | awk '{print $(NF-1)}' | head -n 1 )
jupyter lab --ip=0.0.0.0 --port=8888
