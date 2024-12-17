#!/bin/bash
export CUDA_VISIBLE_DEVICES=`/home/kloudvoj/devel/prompt_optimization/slurm/query_gpus.py`
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)  
echo "NUM_GPUS=${NUM_GPUS}"
echo "VLLM_MY_PORT=${VLLM_MY_PORT}"
MODEL_NAME="$1"
echo "MODEL_NAME=${MODEL_NAME}"
source "/home/kloudvoj/devel/prompt_optimization/slurm/init_environment_vllm_amd.sh"
case "${MODEL_NAME}" in
    "llama70b")
        vllm serve "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --port "${VLLM_MY_PORT}" --gpu-memory-utilization 0.95 --max-model-len 65536 --tensor-parallel-size "${NUM_GPUS}"
        ;;
    "llama70b_3.3")
        vllm serve "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4" --port "${VLLM_MY_PORT}" --gpu-memory-utilization 0.95 --max-model-len 65536 --tensor-parallel-size "${NUM_GPUS}"
        ;;
    "llama3b")
        vllm serve "meta-llama/Llama-3.2-3B-Instruct" --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 30000
        ;;
    "aya8b")
        vllm serve "CohereForAI/aya-expanse-8b" --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 8000
        ;;
    "nemostral")
        vllm serve "mistralai/Mistral-Nemo-Instruct-2407" --port "${VLLM_MY_PORT}" --tokenizer_mode "mistral" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 11000
        ;;
    *)
        echo "ERROR: Unsupported model."
        #vllm serve $1 --model $1 --port "${VLLM_MY_PORT}" --tensor-parallel-size "${NUM_GPUS}" --dtype="half" --gpu-memory-utilization 0.95 --max-model-len 30000
        ;;
esac