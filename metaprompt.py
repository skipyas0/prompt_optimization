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
    def __init__(self, settings: Optional[dict], **kwargs):
        super().__setattr__('settings', settings)
        super().__setattr__('metaprompts', {k: Metaprompt(task=k, **v) for k,v in kwargs["metaprompts"].items()})

    
    def keys(self):
        return self.metaprompts.keys()
    def items(self):
        return self.metaprompts.items()
    def as_dict(self):
        return self.metaprompts
    
    def __getattr__(self, item):
        if item == "settings":
            return self.settings
        if item == "metaprompts":
            return self.metaprompts
        if item in self.metaprompts:
            return self.metaprompts[item]
        raise AttributeError(f"MetapromptSet doesn't include metaprompt '{item}'.")

    def __setattr__(self, key, value):
        if key == "settings":
            # Use super().__setattr__ to set attributes directly
            super().__setattr__('settings', value)
        elif key == "metaprompt":
            super().__setattr__('metaprompts', value)
        else:
            # Update metaprompts dictionary for dynamic attributes
            self.metaprompts[key] = value

    @property
    def traits(self): 
        return filter(lambda x: x.type == "trait", self.metaprompts.values())
