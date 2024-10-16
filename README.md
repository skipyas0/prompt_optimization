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
- try randomly varying mutation prompts with seed phrases to create more original prompts
- rethink metaprompts, standardize syntax - use html-like tags
- rethink template formatting, format brackets are now named to account for persona/style options in template

## Tested models
- hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
- Qwen/Qwen2.5-14B-Instruct
- microsoft/Phi-3.5-mini-instruct (runs on V100)

## Tested datasets
- openai/gsm8k
- microsoft/orca-math-word-problems-200k
- iamollas/ethos (todo)
- maveriq/bigbenchhard (todo)

## Conventions
- Utilize HTML-like tags where possible
- When instructing LLM to fill in the blank, use <-INS-> 'token'
- Formatting brackets in metaprompts are designated
- Prompt is an specimen in the EA and an object. Metaprompts are manually written instructions for the LLM on how to manipulate prompts.
- LLM thinking process/explanation has no syntax demands, but the final answer must be given in double square brackets.