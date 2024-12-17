from typing import Optional, Literal
from utils import FromJSON

class Metaprompt:
    def __init__(self, 
                 task: str, 
                 text: str, 
                 type: Literal["trait", "solve", "evo"],
                 format_ids: set[str]) -> None:
        self.task = task
        self.text = text
        self.type = type
        self.formatting_identifiers = set(format_ids)
        print(f"loaded prompt {self.type}/{self.task} with ins len {len(self.text)}")

    def __str__(self) -> str:
        return self.text
    
    def format(self, replace: dict[str, str]) -> Optional[str]:
        if set(replace.keys()) == self.formatting_identifiers:
            return str(self).format(**replace)
        raise KeyError(f"Formatting failed, keys given: {list(replace.keys())}, keys needed {self.formatting_identifiers}")
    

class MetapromptSet(FromJSON):
    default_path = "conf/_meta/{}.json"
    def __init__(self, name, settings: Optional[dict], **kwargs):
        self.name = name
        self.settings = settings
        self.metaprompts = {k: Metaprompt(task=k, **v) for k,v in kwargs["metaprompts"].items()}
        self.trait_metaprompts = list(filter(lambda x: x.type == "trait", self.metaprompts.values()))
    def keys(self):
        return self.metaprompts.keys()
    def items(self):
        return self.metaprompts.items()
    def as_dict(self):
        return self.metaprompts
    def get(self, key):
        return self.metaprompts[key] if key in self.keys() else None