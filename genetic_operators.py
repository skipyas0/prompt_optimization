from __future__ import annotations
from prompt import Prompt
from typing import Callable


def mutate(prompt: Prompt, mutation_handle: Callable[[list[str]], str]) -> None:
    """
    In-place mutation of prompt trait-by-trait according to self.mutate_trait.
    """
    for ix, trait in enumerate(prompt.traits):
        prompt.traits[ix] = mutation_handle([trait])


def crossover(
    prompt1: Prompt, prompt2: Prompt, crossover_handle: Callable[[list[str]], str]
) -> Prompt:
    """
    Create an offspring by combining this prompt's traits with other prompt
    """
    assert prompt1.n_traits == prompt2.n_traits

    id1 = prompt1.id
    res = prompt1.copy()
    for ix in range(res.n_traits):
        t1, t2 = res.traits[ix], prompt2.traits[ix]
        res.traits[ix] = crossover_handle([t1, t2])

    res.parent_ids = [id1, prompt2.id]
    return res


def de_combination(
    prompts: tuple[Prompt, Prompt, Prompt],
    handles: tuple[
        Callable[[list[str]], str],
        Callable[[list[str]], str],
        Callable[[list[str]], str],
    ],
):
    
    p1, p2, p3 = prompts
    res = p1.copy()

    de1, de2, de3 = handles
    assert res.n_traits == p2.n_traits and res.n_traits == p3.n_traits

    for ix in range(res.n_traits):
        t1, t2, t3 = res.traits[ix], p2.traits[ix], p3.traits[ix]

        diffs = de1([t1, t2])

        mutated = de2([diffs])

        res.traits[ix] = de3([mutated, t3])
    
    res.parent_ids = [p1.id, p2.id, p3.id]
    return res
