from datasets import load_dataset, Dataset
import re
def load_splits(ds_name: str, split: tuple[int, int, int]) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load dataset and split it according to sample numbers in input tuple.
    """
    ds = load_dataset(ds_name, split='train').shuffle(seed=42).select(range(sum(split)))
    infer = ds.select(range(split[0]))
    train = ds.select(range(split[0], sum(split[:2])))
    test = ds.select(range(sum(split[:2]), sum(split)))
    return infer, train, test

def join_dataset_to_str(dataset: Dataset) -> str:   
    """
    Join samples from datasets to single string with <in> <out> html-like tags.
    """

    res = ""

    features = dataset.features()
    for sample in dataset:
        for feature in features:
            res += f"<{feature}> {sample[feature]} </{feature}>\n"
        res += '\n'

    return res

def parse_verdict(text: str) -> str:
    """
    Parse [[[verdict]]]
    """
    pattern = r'\[\[\[(.*?)\]\]\]'
    matches = re.findall(pattern, text)

    return matches