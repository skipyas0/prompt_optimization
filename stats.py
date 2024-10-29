from json import dump
class Stats(dict):
    """
    Wrapper around dict to make keeping statistics through multiple steps easier.
    Just add logging method and call start_step before each EA step.
    """
    def add_to_current_step(self, new_stats: dict[str, int | float]) -> None:
        for stat, new_value in new_stats.items(): 
            if stat not in self.keys():
                self[stat] = [0]
            self[stat][-1] += new_value

    def append_to_current_step(self, new_stats: dict[str, int | float]) -> None:
        for stat, new_value in new_stats.items(): 
            if stat not in self.keys():
                self[stat] = [[]]
            self[stat][-1].append(new_value)

    def join_with_current_step(self, new_stats: dict[str, list[int | float]]) -> None:
        for stat, new_value in new_stats.items(): 
            if stat not in self.keys():
                self[stat] = [new_stats]
            else:
                self[stat][-1] += new_value

    def start_step(self) -> None:
        for value_lists in self.values():
            if type(value_lists[0]) == list:
                value_lists.append([])
            else:
                value_lists.append(0)

    def set_const_stat(self, new_stats: dict[str, int|float]) -> None:
        for stat, value in new_stats.items(): 
            self[stat] = value

    def get_averages(self) -> dict[str, list[int|float]|int|float]:
        return {
            stat: [sum(sublist) / len(sublist) for sublist in value] if type(value[0]) == list else value
            for stat, value in self.items() 
        }
    
    def get_totals(self) -> dict[str, int|float]:
        return {
            stat: sum(value) 
            for stat, value in self.items()
            if type(value[0]) != list
        }
    
    def dump(self, ident: str) -> None:
        with open(f"runs/{ident}/stats.json", 'w') as f:
            dump(self, f, indent=4)
    
stats = Stats()