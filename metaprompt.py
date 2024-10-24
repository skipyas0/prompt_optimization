from typing import Optional

formatting_enforcement_suffixes = {
    'numeric': """After your explanation give your answer as a numeric value in two pairs of square brackets.
<example>
EXPLANATION
[[ANSWER]]
</example>
""",
    'choice': """After your explanation choose the best option and give your answer in two pairs of square brackets.
Answer only with the letter of the option.
<example>
TASK
X: ANSWER1
Y: ANSWER2
EXPLANATION
[[X]]
</example>
""",
'yes-no': """After your explanation write your Yes/No answer into two pairs of square brackets.
Only use 'Yes' or 'No' as your answer. Only terminate your answer with [[Yes]] or [[No]].
""",
'true-false': """After your explanation write your True/False answer into two pairs of square brackets.
Only use 'True' or 'False' as your answer. Only terminate your answer with [[True]] or [[False]].
"""
}

class Metaprompt:
    def __init__(self, task: str, text: str, formatting_identifiers: list[str]) -> None:
        self.text = text
        self.task = task
        self.formatting_identifiers = formatting_identifiers

    def __str__(self) -> str:
        return self.text
    
    def format(self, replace: dict[str, str]) -> Optional[str]:
        if set(replace.keys()) == self.formatting_identifiers:
            return str(self).format(**replace)
        raise KeyError(f"Formatting failed, keys given: {list(replace.keys())}, keys needed {self.formatting_identifiers}")
    
instructions = Metaprompt(
    task="instructions",
    text="""{metapersona}Your mission is to generate instructions for a generic task given a few examples of input/output pairs.
<examples>
{examples}
</examples>
Substitute the <-INS-> tag with your generated instructions. 
Pay attention to the tag's location in the example.
Try to be as concise as possible.
{length}{metastyle}Generated instructions:""",
    formatting_identifiers={"metapersona", "examples", "length", "metastyle"}
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
    formatting_identifiers={"sequence1", "sequence2", "metastyle"}
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
    formatting_identifiers={"sequence", "metastyle"}
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
    formatting_identifiers={"sequence1", "sequence2", "metastyle"}
)

solve = Metaprompt(
    task="solve",
    text="""{preamble}{prefix_instructions}    
    <task>
    {task}
    </task>
    {suffix_instructions}{universal_suffix}
    """,
    formatting_identifiers={"preamble", "prefix_instructions", "task", "suffix_instructions", "universal_suffix"}
)

metaprompts = [
    instructions,
    mutated_crossover,
    mutation,
    crossover,
    solve
]

metaprompts_dict = {
    p.task: p for p in metaprompts
}