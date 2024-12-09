# Prompt 💬 optimization 📈
#### 👷 *Work in progress* 👷
This repository constains code for my bachelor's thesis on the topic of LLM prompt optimalization.
The main focus is on optimization through evolutionary algorithms.

## Usage 📋
As of now only possible using a SLURM job scheduler and a SLURM script on a compute cluster. 
Calls are done with OpenAI API to a VLLM server.
Batch scripts in slurm folder.
#### Two modes
1. Initialization
```
sbatch slurm/run_on_vllm_gpufast1 --conf dataset evoparams model (--run_eval)
```
2. Rerun/run continuation
```
sbatch slurm/run_on_vllm_gpufast1 --ident dataset-evoparams-model-timestamp (--continue_run) (--run_eval)
```
Configuration profiles for dataset, evoparams and model are in the 'conf' subdirectory.

Metaprompt set from 'conf/_meta' is specified in the dataset conf file for each task.

## Contact me 📮
- faculty email: kloudvoj@fel.cvut.cz
- personal email: skipyas0@gmail.com

## TODOs 🤔  

## Tested models 🤖
- hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 (2 A100)
- meta-llama/Llama-3.2-3B-Instruct (1 V100)
- CohereForAI/aya-expanse-8b (1 V100)
- mistralai/Mistral-Nemo-Instruct-2407 (1 V100)
 
## Tested datasets 📚
- openai/gsm8k
- microsoft/orca-math-word-problems-200k
- maveriq/bigbenchhard
- cais/mmlu
- deepmind/code_contests
- livebench

## Conventions
- Utilize HTML-like tags where possible
- When instructing LLM to fill in the blank, use <-INS-> 'token'
- Formatting brackets in metaprompts are designated
- Prompt is an specimen in the EA and an object. Metaprompts are manually written instructions for the LLM on how to manipulate prompts.