from typing import Optional

baseline_suffixes = {
    "Blank": "{}",
    "Kojima": "{}\nLet's think step by step.",  # Kojima et al. 2022
    "Zhou": "{}\nLet's work this out in a step by step way to be sure we have the right answer.",  # Zhou et al. 2022b
    "Fernando": "{}\nSOLUTION:",  # Fernando et al. 2023
}

class Metaprompt:
    def __init__(self, task: str, text: str, formatting_identifiers: list[str]) -> None:
        self.text = text
        self.task = task
        self.formatting_identifiers = formatting_identifiers

    def __str__(self) -> str:
        return self.text
    
    def format(self, replace: dict[str, str]) -> Optional[str]:
        if list(replace.keys()) == self.formatting_identifiers:
            return str(self).format(**replace)
        return None
    
instructions = Metaprompt(
    task="instructions",
    text="""{metapersona}Your mission is to generate instructions for a generic task given a few examples of input/output pairs.
<examples>
{examples}
</examples>
Substitute the <-INS-> tag with your generated instructions. 
Pay attention to the tag's location in the example.
Try to be as concise as possible.
{metastyle}Generated instructions:""",
    formatting_identifiers=["metapersona", "examples", "metastyle"]
)

mutated_crossover = Metaprompt(
    task="mutated_crossover",
    text="""Below you will find two sequences with similar meanings. 
Your task is to take ideas from both sequences and paraphrase them, so that you add novelty.
Avoid using the same words as in the original sequences.

<sequence1>
{sequence1}
</sequence1>

<sequence2>
{sequence2}
</sequence2>
{metastyle}Resulting Sequence:""",
    formatting_identifiers=["sequence1", "sequence2", "metastyle"]
)

mutation = Metaprompt(
    task="mutation",
    text="""Below you will find a sequence. 
Your task is to mutate it - change it, for example by choosing fitting synonyms.
Conserve the original meaning.

<sequence>
{sequence}
</sequence>
{metastyle}Resulting Sequence:""",
    formatting_identifiers=["sequence", "metastyle"]
)

crossover = Metaprompt(
    task="crossover",
    text="""Below you will find two sequences. 
Your task is to identify important aspects of both sequences and then combine them so that the resulting sequence captures the meaning of both sequence.

<sequence1>
{sequence1}
</sequence1>

<sequence2>
{sequence2}
</sequence2>
{metastyle}Resulting Sequence:""",
    formatting_identifiers=["sequence1", "sequence2", "metastyle"]
)

metaprompts = [
    instructions,
    mutated_crossover,
    mutation,
    crossover
]
