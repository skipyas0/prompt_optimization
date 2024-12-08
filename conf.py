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
    def __init__(self, model_api: ModelAPI, task_toolkit: TaskToolkit, evoparams: EvoParams, run_eval: bool, ident: str) -> None:
        self.model_api = model_api
        self.task_toolkit = task_toolkit
        self.evoparams = evoparams
        self.ident = ident
        self.run_eval = run_eval
        self.run_path = self.create_logging_env()
        self.evoparams.prompt_params = PromptParams(self.model_api.solve, 
                                            self.run_path + "{context}/{step_id}.ndjson", 
                                            self.task_toolkit)
    @classmethod                                                
    def from_subconfig_names(cls: Config, model_template: str, task_template: str, evo_template: str, run_eval: bool):
        # used when initializing a new run
        model_api = ModelAPI.from_json(model_template)
        task_toolkit = TaskToolkit.from_json(task_template)
        evoparams = EvoParams.from_json(evo_template)
        time = datetime.now().strftime('%m-%d-%Y_%H-%M-%S_')
        ident = f"{task_toolkit.name}-{evoparams.name}-{model_api.name}-{time}"
        return cls(model_api, task_toolkit, evoparams, run_eval, ident)

    @classmethod                                                
    def from_ident(cls: Config, ident: str, run_eval: bool):
        # used when continuing a run or evaluating
        task_template, evo_template, model_template, time = ident.split('-')
        model_api = ModelAPI.from_json(model_template)
        task_toolkit = TaskToolkit.from_json(task_template)
        evoparams = EvoParams.from_json(evo_template)
        ident = f"{task_toolkit.name}-{evoparams.name}-{model_api.name}-{time}"
        return cls(model_api, task_toolkit, evoparams, run_eval, ident)

    @classmethod
    def from_args(cls: Config) -> Config:
        parser = argparse.ArgumentParser(description="Prompt optimalization with evolutionary algorithm")
        parser.add_argument('model_template', type=str)
        parser.add_argument('task_template', type=str)
        parser.add_argument('evo_template', type=str)
        parser.add_argument('--run_eval', default=False, action='store_true', type=bool)
        args = parser.parse_args()
        return cls.from_subconfig_names(**vars(args))

    def create_logging_env(self) -> str:
        if not os.path.exists(f"runs/{self.ident}"):
            os.mkdir(f"runs/{self.ident}")
            os.mkdir(f"runs/{self.ident}/plots")
            os.mkdir(f"runs/{self.ident}/steps")
            os.mkdir(f"runs/{self.ident}/eval")
        os.environ["CALL_LOG_FILE"] = f"runs/{self.ident}/calls.ndjson"
        return f"runs/{self.ident}/"
