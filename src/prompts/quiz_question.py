class QuizQuestion:
    def __init__(self):
        self.prompts = {
            "quiz_question_prompt": """You are given a five-option multiple-choice quiz.
You have encountered this text during your training. One of the options below is the exact original instance from a real dataset. The other three (A\u2013D) are word-level perturbations where some words have been replaced with contextually similar synonyms. Option E means none of the above.
Your task is to identify which option is the exact original dataset instance by recalling it from your training data.
Important definition of exact: Exact means lexical identity, not semantic similarity. The original option preserves the exact words, phrases, and structure of the real dataset instance.
Rules:
Output only one uppercase letter, \tONLY ONE UPPER CASE LETTER, JUST ONE: A, B, C, D, or E.
DO NOT output any explanation, punctuation, or extra text.
Trust your recognition. If one option feels distinctly more familiar or natural in its exact phrasing, that is the original \u2014 select it.
You may and should draw on your memory of training data. Ask yourself: \"Have I seen this exact phrasing before?\" Use that sense of familiarity as your primary signal.
Output E only as a last resort \u2014 only if you have genuinely zero recognition of any option from any dataset. If you recognize any option at all, output that letter.
Decision policy:
Your goal is not to compare options analytically against a reference text. It is to recall which phrasing you have seen before in training.
If one option feels distinctly more familiar than the others in its exact wording, that is the original.
Use both wording-level comparison and your sense of familiarity with the phrasing.
Do not treat all options as equal variations. One is the real instance \u2014 your training data contains it.
1. Output only one uppercase letter: A, B, C, D, or E.
2. Do not output any explanation, punctuation, or extra text!!!

Options:
Option A)
{option_a}

Option B)
{option_b}

Option C)
{option_c}

Option D)
{option_d}

Option E)
None of the provided options.

Final answer (single letter only):
"""
        }

    def get_prompt(self, prompt_type):
        return self.prompts.get(prompt_type, "Invalid prompt type")
