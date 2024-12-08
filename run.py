from conf import Config
import data_evaluation as eval
from evolutionary import EvolutionaryAlgorithm
config = Config.from_args()
EA = EvolutionaryAlgorithm(config)
EA.populate()
EA.run()

if config.run_eval:
    gens = EA.population_through_steps
    baselines = EA.random_baseline
    EV = eval.Eval(gens, baselines, config)
    EV.run()