#!/bin/bash
export CUDA_VISIBLE_DEVICES=`/home/kloudvoj/devel/universal_scripts/query_gpus.py`
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)  
echo "NUM_GPUS=${NUM_GPUS}"
echo "VLLM_MY_PORT=${VLLM_MY_PORT}"

source /home/kloudvoj/devel/prompt_optimization/slurm/init_environment_vllm_amd.sh
case "$1" in
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
        vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --port "${VLLM_MY_PORT}" --gpu-memory-utilization 0.95 --max-model-len 65536 --tensor-parallel-size "${NUM_GPUS}"
        ;;
    "microsoft/Phi-3.5-mini-instruct")
        vllm serve "microsoft/Phi-3.5-mini-instruct" --model "microsoft/Phi-3.5-mini-instruct" --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 30000
        ;;
    *)
        echo "ERROR: Unsupported model."
        #vllm serve $1 --model $1 --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 30000
        ;;
esac