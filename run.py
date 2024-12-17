from conf import Config
import data_evaluation as eval
from evolutionary import EvolutionaryAlgorithm
from time import time
from stats import stats

t0 = time()

config = Config.from_args()
EA = EvolutionaryAlgorithm(config)

t1 = time()

EA.run()

t2 = time()

if config.run_eval:
    gens = EA.population_through_steps
    baselines = EA.random_baseline
    EV = eval.Eval(gens, baselines, config)
    EV.run()

t3 = time()

stats.set_const_stat({
    "Init time": t1 - t0,
    "Run time": t2 - t1,
    "Eval time": t3 - t2,
    "Total time": t3 - t0
})

stats.save(config.ident)
print(f"Finished run with ident\n{config.ident}\nafter {round((t3-t0)//60)}m{round((t3-t0)%60)}s")