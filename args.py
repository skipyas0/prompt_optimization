import argparse
from evolutionary import EvoParams
from vllm_api import OpenAIPredictor
from utils import load_splits
from datasets import Dataset
from types import NoneType

def parse_args(log_file: str):
    parser = argparse.ArgumentParser(description="Prompt optimalization with evolutionary algorithm")
    parser.add_argument('--initial_population_size', type=int, default=20, help='Initial size of the population')
    parser.add_argument('--population_change_rate', type=int, default=0, help='Rate of population change')
    parser.add_argument('--mating_pool_size', type=int, default=10, help='Size of the mating pool')
    parser.add_argument('--trait_ids', type=str, nargs='+', default=["instructions"], help='List of trait identifiers')
    parser.add_argument('--task_insert_ix', type=int, default=1, help='Position where to insert task into traits')
    parser.add_argument('--prompt_mutation_probability', type=float, default=1.0, help='Probability of prompt mutation')
    parser.add_argument('--trait_mutation_percentage', type=float, default=1.0, help='Portion of traits mutated when mutating prompt specimen')
    parser.add_argument('--max_iters', type=int, default=20, help='Maximum number of iterations')
    parser.add_argument('--evolution_mode', type=str, choices=['GA', 'DE'], default='GA', help="Mode of evolution: 'GA' or 'DE'")
    parser.add_argument('--selection_mode', type=str, choices=['rank', 'roulette', 'tournament'], default='roulette', help="Selection mode: 'rank', 'roulette', or 'tournament'")
    parser.add_argument('--tournament_group_size', type=int, default=3, help='Size of the tournament group for tournament selection')
    parser.add_argument('--train_batch_size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--log', action='store_true', default=True, help='Enable logging (default: True)')
    parser.add_argument('--no-log', action='store_false', dest='log', help='Disable logging')

    parser.add_argument('--ds', type=str, default="microsoft/orca-math-word-problems-200k", help='Dataset name')
    parser.add_argument('--split', type=int, nargs=3, default=[4, 100, 30], help='Split sizes for initial generatIion, training and evaluation sets')
    parser.add_argument('--model', type=str, default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", help='Model used for generation')
    parser.add_argument('--temp', type=float, default=0.75, help='Temperature for model sampling')

    parser.add_argument('--debug', action='store_true', help='Debug mode: No LLM, go through evolution with scrambling text and assigning random scores.')
    args = parser.parse_args()

    if args.log:
        from json import dump
        with open(log_file, 'w') as f:
            dump(vars(args), f, indent=4)
    return args

def parse_args_and_init(log_file: str) -> tuple[EvoParams, tuple[Dataset, Dataset, Dataset], OpenAIPredictor | NoneType]:
    """
    Parses CLI args and initialized parameters for evolutionary algorithm, dataset splits and OpenAI API object.
    """
    args = parse_args(log_file)
    evo_params = EvoParams(
        initial_population_size=args.initial_population_size,
        population_change_rate=args.population_change_rate,
        mating_pool_size=args.mating_pool_size,
        trait_ids=args.trait_ids,
        task_insert_ix=args.task_insert_ix,
        prompt_mutation_probability=args.prompt_mutation_probability,
        trait_mutation_percentage=args.trait_mutation_percentage,
        max_iters=args.max_iters,
        evolution_mode=args.evolution_mode,
        selection_mode=args.selection_mode,
        tournament_group_size=args.tournament_group_size,
        train_batch_size=args.train_batch_size,
        log=args.log
    )
    if args.debug:
        api = None
    else:
        api = OpenAIPredictor(args.model, args.temp)
    return evo_params, load_splits(args.ds, args.split), api

    
