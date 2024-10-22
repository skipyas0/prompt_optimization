sbatch slurm/run_on_vllm_amdgpufast.bash --initial_population_size 10 --mating_pool_size 6 --prompt_mutation_probability 0.5 --max_iters 10 --train_batch_size 3 --temp 0.5

sbatch slurm/run_on_vllm_amdgpufast.bash "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --ds="openai/gsm8k" --train_batch_size=8 --combine_co_mut --scorer="binary_match"


# reduced llm calls
sbatch slurm/run_on_vllm_gpufast1.bash "microsoft/Phi-3.5-mini-instruct" --ds="openai/gsm8k" --train_batch_size=8 --combine_co_mut --scorer="binary_match" --initial_population_size 12 --mating_pool_size 6 --max_iters 10 --temp 0.5

sbatch slurm/run_on_vllm_gpufast4.bash "Qwen/Qwen2.5-14B-Instruct" --ds="openai/gsm8k" --train_batch_size=8 --combine_co_mut --scorer="binary_match" --initial_population_size 12 --mating_pool_size 6 --max_iters 10 --temp 0.5

sbatch slurm/run_on_vllm_amdgpufast2.bash "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --ds="openai/gsm8k" --train_batch_size=6 --combine_co_mut --scorer="binary_match" --initial_population_size 10 --mating_pool_size 6 --max_iters 8 --temp 0.5

# similarity filtering

sbatch slurm/run_on_vllm_amdgpufast2.bash "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --ds="openai/gsm8k" --train_batch_size=6 --combine_co_mut --scorer="binary_match" --initial_population_size 10 --mating_pool_size 6 --max_iters 8 --temp 0.5 --filter_similar_method="bert" --filter_th 0.9 --repop_method_proportion 0.8

sbatch slurm/run_on_vllm_gpufast1.bash "microsoft/Phi-3.5-mini-instruct" --ds="openai/gsm8k" --train_batch_size=8 --combine_co_mut --scorer="binary_match" --initial_population_size 12 --mating_pool_size 6 --max_iters 10 --temp 0.5 --filter_similar_method="bert" --filter_th 0.9 --repop_method_proportion 0.8

# metapersona + metastyles

sbatch slurm/run_on_vllm_amdgpufast2.bash "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --ds="openai/gsm8k" --train_batch_size=6 --combine_co_mut --scorer="binary_match" --initial_population_size 10 --mating_pool_size 6 --max_iters 8 --temp 0.5 --filter_similar_method="bert" --filter_th 0.9 --repop_method_proportion 0.8 --metapersonas --metastyles