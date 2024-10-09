baseline_suffixes = [
    "",
    "Let's think step by step."  # Kojima et al. 2022
    "Let's work this out in a step by step way to be sure we have the right answer."  # Zhou et al. 2022b
    "SOLUTION:",  # Fernando et al. 2023
]

prompt_suffix_for_usage = """
    After your explanation, make sure you put your final answer in two pairs of square brackets.
    <example>
    ...
    And for the above reasons, the solution is ANSWER.
    [[ANSWER]]
    </example>
    """

metaprompts = {
    "static": {
        "crossover": 
"""Below you will find two sequences. 
Your task is to identify important aspects of both sequences and then combine them so that the resulting sequence captures the meaning of both sequence.
Sequence 1:
{}
End of Sequence 1
Sequence 2:
{}
End of Sequence 2
Resulting Sequence:""",

        "de1": """
    Below you will find two sequences.
Your task is to identify important parts of both sentences in which they differ.
Output these differing parts as a list of comma-seperated words.

Sequence 1:
{}
End of Sequence 1

Sequence 2:
{}
End of Sequence 2

Resulting Sequence:
    """,
        "de2": """
    Below you will find a sequence of comma-seperated words.
Your tasks is to mutate each word - find good synonyms.
Output them as a list of comma-seperated values.

Sequence 1:
{}
End of Sequence 1

Resulting Sequence:
    """,
        "de3": """
    Below you will find two sequences.
Sequence 1 is a list of words seperated by commas.
Your task is to take those words and encorporate them into Sequence 2.
Add the meaning of the words to the original sequence.

Sequence 1
{}
End of Sequence 1

Sequence 2
{}
End of Sequence 2

Resulting Sequence:
    """,
        "instructions": """
    You are a coach capable of directing people to solve any task.
Below are question and answer pairs pertaining to a task.
Give specific and detailed instruction to a client looking to solve a task similar to these examples.
Your instructions should maximize the likelihood of correct answers for similar questions.

Examples
{}
End of examples

You are replacing <-INS-> with your answer. Get straight to the point.
    """,
        "mutated_crossover": """
    Below you will find two sequences with similar meanings. 
Your task is to take ideas from both sequences and paraphrase them, so that you add novelty.
Avoid using the same words as in the original sequences.

Sequence 1:
{}
End of Sequence 1

Sequence 2:
{}
End of Sequence 2

Resulting Sequence:
    """,
        "mutation": """
    Below you will find a sequence. 
Your task is to mutate it - change it, for example by choosing fitting synonyms.
Conserve the original meaning.

Sequence
{}
End of Sequence

Resulting Sequence:
    """,
    },
    "seeded": {},
}
