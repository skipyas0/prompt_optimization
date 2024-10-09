# Prompt optimalization
This repository constains code for my bachelor's thesis on the topic of LLM prompt optimalization.
The main focus is on optimalization through evolutionary algorithms.

## Usage
As of now only possible using a SLURM job scheduler and a SLURM script on a compute cluster. 
Calls are done with OpenAI API to a VLLM server.
Batch scripts in slurm folder.
Mandatory argument: model
Example:
'''
sbatch slurm/run_on_vllm_amdgpufast.bash "microsoft/Phi-3.5-mini-instruct" --initial_population_size 10
'''

Debug mode: *python run.py --debug*

## Contact me
- faculty email: kloudvoj@fel.cvut.cz
- personal email: skipyas0@gmail.com

## TODOs
- add ability to run on OpenAI cloud
- add support to run on local GPU with some models
- add other implementations of evolutionary algorithms - currently only classic genetic algorithm and differential evolution are supported
- try randomly varying mutation prompts with seed phrases to create more original prompts
- rethink metaprompts, standardize syntax

## Tested models
- hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
- Qwen/Qwen2.5-14B-Instruct
- microsoft/Phi-3.5-mini-instruct (runs on V100)

## Tested datasets
- openai/gsm8k
- microsoft/orca-math-word-problems-200k
- iamollas/ethos
- maveriq/bigbenchhard