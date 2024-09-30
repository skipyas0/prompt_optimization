sbatch slurm/run_on_vllm_amdgpufast.bash --initial_population_size 10 --mating_pool_size 6 --prompt_mutation_probability 0.5 --max_iters 10 --train_batch_size 3 --temp 0.5

sbatch slurm/run_on_vllm_amdgpufast.bash "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --ds="openai/gsm8k" --train_batch_size=8 --combine_co_mut --scorer="binary_match"