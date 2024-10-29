#!/bin/bash
export CUDA_VISIBLE_DEVICES=`/home/kloudvoj/devel/universal_scripts/query_gpus.py`
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)  
echo "NUM_GPUS=${NUM_GPUS}"
echo "VLLM_MY_PORT=${VLLM_MY_PORT}"

source /home/kloudvoj/devel/prompt_optimization/slurm/init_environment_vllm_amd.sh
case "$1" in
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
        vllm serve $1 --model $1 --port "${VLLM_MY_PORT}" --gpu-memory-utilization 0.95 --max-model-len 65536 --tensor-parallel-size "${NUM_GPUS}"
        ;;
    "microsoft/Phi-3.5-mini-instruct")
        vllm serve $1 --model $1 --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 30000
        ;;
    "meta-llama/Llama-3.2-3B-Instruct")
        vllm serve $1 --model $1 --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 30000
        ;;
    "CohereForAI/aya-expanse-8b")
        vllm serve $1 --model $1 --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 8000
        ;;
    "mistralai/Mistral-Nemo-Instruct-2407")
        vllm serve $1 --model $1 --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 11000
        ;;
    *)
        echo "ERROR: Unsupported model."
        #vllm serve $1 --model $1 --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 30000
        ;;
esac