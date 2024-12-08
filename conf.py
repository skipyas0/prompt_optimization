from __future__ import annotations
import os
from utils import FromJSON
from dataset_loader import TaskToolkit
from model_api import ModelAPI
from evolutionary import EvoParams
import argparse
from datetime import datetime
from prompt import PromptParams
class Config(FromJSON):
    def __init__(self, task_toolkit: TaskToolkit, evoparams: EvoParams, model_api: ModelAPI, run_eval: bool, continue_run: bool, ident: str) -> None:
        self.model_api = model_api
        self.task_toolkit = task_toolkit
        self.evoparams = evoparams
        self.ident = ident
        self.run_eval = run_eval
        self.run_path = self.create_logging_env()
        self.continue_run = continue_run
        if self.model_api.model == "debug":
            from random import random
            self.task_toolkit.scoring_function = lambda _1, _2: random()
        self.evoparams.prompt_params = PromptParams(self.model_api.solve, 
                                            self.run_path + "{context}/{step_id}.ndjson", 
                                            self.task_toolkit)
        
    @classmethod                                                
    def from_subconfig_names(cls: Config, task_template: str, evo_template: str, model_template: str,  run_eval: bool, continue_run: bool):
        # used when initializing a new run
        task_toolkit = TaskToolkit.from_json(task_template)
        evoparams = EvoParams.from_json(evo_template)
        model_api = ModelAPI.from_json(model_template)
        time = datetime.now().strftime('%m.%d.%Y_%H:%M:%S')
        ident = f"{task_toolkit.name}-{evoparams.name}-{model_api.name}-{time}"
        return cls(task_toolkit, evoparams, model_api, run_eval, continue_run, ident)

    @classmethod                                                
    def from_ident(cls: Config, ident: str, run_eval: bool, continue_run: bool):
        # used when continuing a run or evaluating
        task_template, evo_template, model_template, _ = ident.split('-')
        task_toolkit = TaskToolkit.from_json(task_template)
        evoparams = EvoParams.from_json(evo_template)
        model_api = ModelAPI.from_json(model_template)
        return cls(task_toolkit, evoparams, model_api, run_eval, continue_run, ident)

    @classmethod
    def from_args(cls: Config) -> Config:
        parser = argparse.ArgumentParser(description="Prompt optimalization with evolutionary algorithm")
        parser.add_argument('--conf', type=str, nargs=3, default=None)
        parser.add_argument('--ident', type=str, default=None)
        parser.add_argument('--run_eval', default=False, action='store_true')
        parser.add_argument('--continue_run', default=False, action='store_true')
        args = parser.parse_args()
        if args.conf:
            task, evo, model = args.conf
            return cls.from_subconfig_names(task, evo, model, args.run_eval, args.continue_run)
        if args.ident:
            return cls.from_ident(args.ident, args.run_eval, args.continue_run)
        raise argparse.ArgumentError("Either --conf or --ident has to be defined.")
            

    def create_logging_env(self) -> str:
        if not os.path.exists(f"runs/{self.ident}"):
            os.mkdir(f"runs/{self.ident}")
            os.mkdir(f"runs/{self.ident}/steps")
        os.environ["CALL_LOG_FILE"] = f"runs/{self.ident}/calls.ndjson"
        return f"runs/{self.ident}/"
