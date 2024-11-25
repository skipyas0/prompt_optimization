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
    def __init__(self, model_api: str, task_toolkit: str, evoparams: str, name: str) -> None:
        self.model_api = ModelAPI.from_json(model_api)
        self.task_toolkit = TaskToolkit.from_json(task_toolkit)
        self.evoparams = EvoParams.from_json(evoparams)
        self.name = name
        self.run_path = self.create_logging_env()
        self.evoparams.prompt_params = PromptParams(self.model_api.solve, 
                                            self.run_path + "steps/step{}.ndjson", 
                                            self.task_toolkit)

    @classmethod
    def from_args(cls: Config) -> Config:
        parser = argparse.ArgumentParser(description="Prompt optimalization with evolutionary algorithm")
        parser.add_argument('model_template', type=str)
        parser.add_argument('task_template', type=str)
        parser.add_argument('evo_template', type=str)
        args = parser.parse_args()
        return cls(**args)

    def create_logging_env(self) -> str:
        self.time = datetime.now().strftime('%m-%d-%Y_%H-%M-%S_')
        self.ident = f"{self.task_toolkit.name}-{self.evoparams.name}-{self.model_api.name}-{self.time}"
        os.mkdir(f"runs/{self.ident}")
        os.mkdir(f"runs/{self.ident}/plots")
        os.mkdir(f"runs/{self.ident}/steps")
        return f"runs/{self.ident}/"
