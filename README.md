# Prompt optimalization
#### *Work in progress*
This repository constains code for my bachelor's thesis on the topic of LLM prompt optimalization.
The main focus is on optimalization through evolutionary algorithms.

## Usage
As of now only possible using a SLURM job scheduler and a SLURM script on a compute cluster. 
Calls are done with OpenAI API to a VLLM server.
Batch scripts in slurm folder.
Mandatory argument: model
Example:
```
sbatch slurm/run_on_vllm_amdgpufast.bash "microsoft/Phi-3.5-mini-instruct" --initial_population_size 10
```

Debug mode: *python run.py --debug*

## Contact me
- faculty email: kloudvoj@fel.cvut.cz
- personal email: skipyas0@gmail.com

## TODOs
- implement checkpoints to run longer evolutions in more compute sessions, redesign logging to capture all info for continuing the run
- create stronger baselines

## Tested models
- hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 (2 A100)
- Qwen/Qwen2.5-14B-Instruct (1 A100)
- microsoft/Phi-3.5-mini-instruct (1 V100)
- meta-llama/Llama-3.2-3B-Instruct (1 V100)
- CohereForAI/aya-expanse-8b (1 V100)
- mistralai/Mistral-Nemo-Instruct-2407 (1 V100)

## Tested datasets
- openai/gsm8k
- microsoft/orca-math-word-problems-200k
- maveriq/bigbenchhard
- GBaker/MedQA-USMLE-4-options
- cais/mmlu

## Conventions
- Utilize HTML-like tags where possible
- When instructing LLM to fill in the blank, use <-INS-> 'token'
- Formatting brackets in metaprompts are designated
- Prompt is an specimen in the EA and an object. Metaprompts are manually written instructions for the LLM on how to manipulate prompts.
- LLM thinking process/explanation has no syntax demands, but the final answer must be given in double square brackets.