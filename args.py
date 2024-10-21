import argparse
from evolutionary import EvoParams
import utils
from datasets import Dataset
from typing import Optional
from vllm_api import OpenAIPredictor
from local_api import LocalPredictor

def parse_args(log_file: str):
    parser = argparse.ArgumentParser(description="Prompt optimalization with evolutionary algorithm")
    parser.add_argument('model', type=str, help='Model used for generation (huggingface link)')
    
    parser.add_argument('--initial_population_size', type=int, default=6, help='Initial size of the population')
    parser.add_argument('--population_change_rate', type=int, default=0, help='Rate of population change')
    parser.add_argument('--mating_pool_size', type=int, default=4, help='Size of the mating pool')
    parser.add_argument('--trait_ids', type=list[str], nargs='+', default=["instructions"], help='List of trait identifiers')
    parser.add_argument('--task_insert_ix', type=int, default=1, help='Position where to insert task into traits')
    parser.add_argument('--prompt_mutation_probability', type=float, default=0.5, help='Probability of prompt mutation')
    parser.add_argument('--trait_mutation_percentage', type=float, default=1.0, help='Portion of traits mutated when mutating prompt specimen')
    parser.add_argument('--max_iters', type=int, default=8, help='Maximum number of iterations')
    parser.add_argument('--evolution_mode', type=str, choices=['GA', 'DE'], default='GA', help="Mode of evolution: 'GA' or 'DE'")
    parser.add_argument('--selection_mode', type=str, choices=['rank', 'roulette', 'tournament'], default='rank', help="Selection mode: 'rank', 'roulette', or 'tournament'")
    parser.add_argument('--tournament_group_size', type=int, default=3, help='Size of the tournament group for tournament selection')
    parser.add_argument('--train_batch_size', type=int, default=3, help='Batch size for training')
    parser.add_argument('--log', action='store_true', default=True, help='Enable logging (default: True)')
    parser.add_argument('--no-log', action='store_false', dest='log', help='Disable logging')
    parser.add_argument('--combine_co_mut', action='store_true', default=False, help='Instead of crossover and mutation, perform only crossover with a metaprompt with more emphasis on paraphrasing.')
    parser.add_argument('--ds', type=str, default="microsoft/orca-math-word-problems-200k", help='Dataset name')
    parser.add_argument('--split', type=int, nargs=3, default=[4, 100, 20], help='Split sizes for initial generatIion, training and evaluation sets')
    parser.add_argument('--scorer', type=str, choices=['ask_llm_to_compare', 'levenshtein', 'binary_match', 'rouge', 'bert'],default="ask_llm_to_compare", help='Function used for evaluation of result')
    parser.add_argument('--temp', type=float, default=0.5, help='Temperature for model sampling')
    parser.add_argument('--local', action='store_true', help='Local mode: Instead of calling a VLLM server use a transformers pipeline.')
    parser.add_argument('--debug', action='store_true', help='Debug mode: No LLM, go through evolution with scrambling text and assigning random scores.')
    parser.add_argument('--filter_similar_method', type=str, choices=['None', 'bert', 'levenshtein', 'rouge'], default='None',help='How to filter prompts based on similarity. If not None, it is applied before selection mechanism.')
    parser.add_argument('--filter_th', type=float, default=0.95, help='Filtration threshold - prompts with higher similarity are deduplicated.')
    parser.add_argument('--repop_method_proportion', type=float, default=1.0, help='Probability of using lamarckian mutation (creating fresh prompts like in initial population) instead of mutating remaining prompts when pop_size<mating_pool_size (too many prompts similarity-filtered).')
    parser.add_argument('--metapersonas', action='store_true', default=False, help='Choose a random metapersona to aid in prompt diversity.')
    parser.add_argument('--metastyles', action='store_true', default=False, help='Add a random thinking/wording style to LLM calls to promote diversity.')

    args = parser.parse_args()

    if args.log:
        from json import dumps
        args_dict = vars(args)
        args_dict["type"] = "args"
        with open(log_file, 'w') as f:
            f.write(dumps(vars(args)) + '\n')
    return args

def parse_args_and_init(log_file: str) -> tuple[EvoParams, tuple[Dataset, Dataset], Optional[OpenAIPredictor | LocalPredictor]]:
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
        log=args.log,
        combine_co_mut=args.combine_co_mut,
        scorer=args.scorer,
        filter_similar_method=args.filter_similar_method,
        filter_th=args.filter_th,
        repop_method_proportion=args.repop_method_proportion,
        metapersonas=args.metapersonas,
        metastyles=args.metastyles
    )
    if args.debug:
        api = None
    else:
        api = LocalPredictor(args.model, args.temp) if args.local else OpenAIPredictor(args.model, args.temp)

    infer, train, eval_data = utils.load_splits(args.ds, args.split)

    instruction_insertion_token = "<-INS->\n"
    examples = utils.join_dataset_to_str(infer, instruction_insertion_token)
    evo_params.examples_for_initial_generation = examples
    return evo_params, (train, eval_data) ,api

    
