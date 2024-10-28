#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=amdgpu --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --out=/home/kloudvoj/devel/prompt_optimization/logs/vllm-api.%j.out
#SBATCH --job-name evoprompt-run
#SBATCH --mail-user=kloudvoj@fel.cvut.cz

source "/home/kloudvoj/devel/prompt_optimization/slurm/init_environment_vllm_amd.sh"

# First run the VLLM server, so we can use OpenAI API
# output redirection however does not work well...

#export OUTLINES_CACHE_DIR=/tmp/.kloudvoj.vlll.outlines
export VLLM_MY_PORT=$(shuf -i8000-8999 -n1)
echo "VLLM_MY_PORT=${VLLM_MY_PORT}"

export VLLM_LOG="/home/kloudvoj/devel/prompt_optimization/logs/vllm-api.$SLURM_JOB_ID.vllm_server.out"
nohup /home/kloudvoj/devel/prompt_optimization/slurm/vllm-serve.bash $1 2>&1 > "$VLLM_LOG" &

# Wait for VLLM server startup
check_substring() {
  grep -q "Avg prompt throughput:" "$VLLM_LOG"
}
while ! check_substring; do
  echo "Waiting for VLLM to start..."
  sleep 3
done

# Now run code which uses the server
export PYTHONPATH=.:$PYTHONPATH
python /home/kloudvoj/devel/prompt_optimization/run.py $*