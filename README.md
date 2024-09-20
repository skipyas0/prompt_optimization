# Prompt optimalization
This repository constains code for my bachelor's thesis on the topic of LLM prompt optimalization.
The main focus is on optimalization through evolutionary algorithms.

## Usage
As of now only possible using a SLURM job scheduler and a SLURM script on a compute cluster. 
Calls are done with OpenAI API to a VLLM server.
Batch scripts in slurm folder.
Example:
'''
sbatch slurm/run_on_vllm_amdgpufast.bash --initial_population_size 10
'''

Debug mode: *python run.py --debug*

## Contact me
- faculty email: kloudvoj@fel.cvut.cz
- personal email: skipyas0@gmail.com

## TODOs
- add ability to run on OpenAI cloud
- add support to run on local GPU with some models
- find other compatible models and add them
- add other implementations of evolutionary algorithms - currently only classic genetic algorithm and differential evolution are supported