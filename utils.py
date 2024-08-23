from random import shuffle
from typing import Any

def random_equal_split(l: list[Any]) -> tuple[list[Any], list[Any]]:
    if len(l) % 2 != 0:
        l = l[:-1]
    half = len(l) // 2
    shuffled = shuffle(l)
    a = shuffled[:half]
    b = shuffled[half+1:]
    return (a,b)