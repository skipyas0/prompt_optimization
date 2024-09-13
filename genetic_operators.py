from __future__ import annotations
from prompt import Prompt
from typing import Callable


def mutate(prompt: Prompt, mutation_handle: Callable[[list[str]], str]) -> None:
    """
    In-place mutation of prompt trait-by-trait according to self.mutate_trait.
    """
    for ix, trait in enumerate(prompt.traits):
        prompt[ix] = mutation_handle[[trait]]


def crossover(
    prompt1: Prompt, prompt2: Prompt, crossover_handle: Callable[[list[str]], str]
) -> Prompt:
    """
    Create an offspring by combining this prompt's traits with other prompt
    """
    assert prompt1.n_traits == prompt2.n_traits

    prompt1 = prompt1.copy()
    for ix in range(prompt1.n_traits):
        t1, t2 = prompt1[ix], prompt2[ix]
        prompt1[ix] = crossover_handle([t1, t2])

    return prompt1


def de_combination(
    prompts: tuple[Prompt, Prompt, Prompt],
    handles: tuple[
        Callable[[list[str]], str],
        Callable[[list[str]], str],
        Callable[[list[str]], str],
    ],
):
    
    p1, p2, p3 = prompts
    p1 = p1.copy()

    de1, de2, de3 = handles
    assert p1.n_traits == p2.n_traits and p1.n_traits == p3.n_traits

    for ix in range(p1.n_traits):
        t1, t2, t3 = p1[ix], p2[ix], p3[ix]

        diffs = de1([t1, t2])

        mutated = de2([diffs])

        p1.traits[ix] = de3([mutated, t3])

    return p1
