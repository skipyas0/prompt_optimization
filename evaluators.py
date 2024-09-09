def simple_list_intersection(ground: list, sample: list) -> float:
    """
    Outputs number of common (exactly equal in both lists) elements divided by the number of total unique elements.
    """
    sg = set(ground)
    ss = set(sample)
    common = sg & ss
    total = sg | ss
    return len(common) / len(total)