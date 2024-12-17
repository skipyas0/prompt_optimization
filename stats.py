from json import dump
import itertools
import matplotlib.pyplot as plt
import os

class Stats(dict):
    """
    Wrapper around dict to make keeping statistics through multiple steps easier.
    Just add logging method and call start_step before each EA step.
    """
    def add_to_current_step(self, new_stats: dict[str, int | float]) -> None:
        for stat, new_value in new_stats.items(): 
            #print("{stat}: {new_value}")
            if stat not in self.keys():
                self[stat] = [0]
            self[stat][-1] += new_value

    def append_to_current_step(self, new_stats: dict[str, int | float]) -> None:
        for stat, new_value in new_stats.items(): 
            #print("{stat}: {new_value}")
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
            stat: value if type(value) != list else [sum(sublist) / len(sublist) for sublist in value] if type(value[0]) == list else value
            for stat, value in self.items() 
        }
    
    def get_totals(self) -> dict[str, int|float]:
        return {
            stat: sum(value) 
            for stat, value in self.items()
            if type(value[0]) != list
        }
    
    def save(self, ident: str) -> None:
        i=1
        while os.path.exists("runs/{ident}/stats{i}.json"):
            i+1

        with open(f"runs/{ident}/stats{i}.json", 'w') as f:
            dump(self.get_averages(), f, indent=4)
        self.plot_training_stats(ident, i)

    def plot_training_stats(self, ident: str, run_num: int) -> None:
        color_cycle = itertools.cycle(plt.cm.get_cmap('tab10').colors)
    
        for name, values in self.get_averages().items():
            if type(values) != list:
                continue
            plt.figure()
            color = next(color_cycle)
            generations = range(1, len(values) + 1) 
            plt.plot(generations, values, label=name, color=color)
    
            plt.ylabel(f'{name}')
            plt.xlabel('Step')
            plt.title(f'Progress of {name.lower()} in training')
            
            plt.legend()
            plt.savefig(f'runs/{ident}/plots/{name.lower().replace(" ", "_")}{run_num}.svg', format='svg')
    
stats = Stats()