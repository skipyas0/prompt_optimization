from __future__ import annotations
import utils
from prompt import Prompt
import matplotlib.pyplot as plt
import os
import json
from conf import Config
import argparse

class EvalResults(utils.FromJSON):
    default_path = "runs/{}"
    def __init__(self, batch, gens, baseline, gens_stats, baseline_stats):
        self.batch = batch.to_dict()
        self.gens = gens
        self.baseline = baseline
        self.gens_stats = gens_stats
        self.baseline_stats = baseline_stats

class Eval():
    def __init__(self, generations: list[list[Prompt]], baseline: list[Prompt], config: Config) -> None:
        self.generations = generations
        self.baseline = baseline
        self.all_prompts = [prompt for step_pop in self.generations for prompt in step_pop]
        self.config = config
        self.test_data = config.task_toolkit.splits.test

    @classmethod
    def from_files(cls: Eval, ident: str) -> Eval:
        generations = []
        config = Config.from_ident(ident, True, False)
        steps_path = f"runs/{ident}/steps"
        files = list(sorted(os.listdir(steps_path), reverse=True))
        baseline = None
        for fn in files:
            with open(steps_path+f"/{fn}", "r") as f:
                gen = []
                lines = f.readlines()
                for l in lines:
                    p = Prompt(json.loads(l), config.evoparams.prompt_params)
                    gen.append(p)
                if fn == "baseline.ndjson":
                    baseline = gen.copy()
                else:
                    generations.append(gen)
        return Eval(generations, baseline, config)
    
    @classmethod
    def from_args(cls: Eval) -> Eval:
        parser = argparse.ArgumentParser(description="Evaluate previous optimization run, input ident")
        parser.add_argument('--ident', type=str, default=None)
        args = parser.parse_args()
        if args.ident:
            return cls.from_files(args.ident)
        raise argparse.ArgumentError("--ident has to be defined.")
    
    def get_fitness_data(self) -> tuple[list[list[float]], list[float]] :
        gen_perfs = [[(p.id, p.calculate_fitness(self.test_data)) for p in g] for g in self.generations]
        baseline_perfs = [p.calculate_fitness(self.test_data) for p in self.baseline]
        return gen_perfs, baseline_perfs    
    
    def run(self):
        gens, baseline = self.get_fitness_data()
        gens_stats = [utils.seq_stats([perf_w_id[1] for perf_w_id in gen_perf]) for gen_perf in gens]
        baseline_stats = utils.seq_stats(baseline)
        results = EvalResults(self.test_data, gens, baseline, gens_stats, baseline_stats)
        results.to_json(f"{self.config.ident}/eval_results.json")
        self.plot([[perf_w_id[1] for perf_w_id in gen_perf] for gen_perf in gens], gens_stats, baseline_stats)


    def plot(self, gens, gens_stats, baseline_stats) -> None:
        plt.figure()
        c1, c2 = plt.cm.get_cmap('tab10').colors[:2]
        data_len = len(gens_stats)
        
        # gens
        gen_nums = range(1, data_len + 1) 
        means = [g.mean for g in gens_stats]
        medians = [g.median for g in gens_stats]
        mins = [g.min for g in gens_stats]
        maxs = [g.max for g in gens_stats]
        plt.plot(gen_nums, means, color=c1, linestyle='-', label=f'Generations mean', alpha=0.5)
        plt.plot(gen_nums, medians, color=c1, linestyle='-.', label=f'Generations median', alpha=0.5)
        plt.fill_between(gen_nums, mins, maxs, color=c1, alpha=0.2)
        for gi, gen in enumerate(gens):
            plt.plot([gi+1]*len(gen), gen, 'x', color=c1, alpha=0.3) 
            
        # baseline
        plt.axhline(baseline_stats.mean, color=c2, linestyle='-', label=f'Baseline mean', alpha=0.5)
        plt.axhline(baseline_stats.median, color=c2, linestyle='-.', label=f'Baseline median', alpha=0.5)
        plt.fill_between(gen_nums, baseline_stats.min, baseline_stats.max, color=c2, alpha=0.2)

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution progress')
        plt.legend()  
        plt.savefig(f'runs/{self.config.ident}/plots/results.svg', format='svg')

if __name__ == "__main__":
    EV = Eval.from_args()
    EV.run()