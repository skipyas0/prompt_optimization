#!/bin/bash
#SBATCH --partition gpufast
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 128G
#SBATCH --gres gpu:1
#SBATCH --time 4:00:00
#SBATCH --job-name evoprompt-test
#SBATCH --output /home/kloudvoj/devel/prompt_optimalization/logs/slurm_out/notebook-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@login.rci.cvut.cz

Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: login.rci.cvut.cz
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

ml Python/3.11.5-GCCcore-13.2.0
source /home/kloudvoj/devel/venv/base/bin/activate
export PATH=/home/kloudvoj/devel/venv/base/bin:${PATH}

jupyter-notebook --no-browser --port=${port} --ip=${node}