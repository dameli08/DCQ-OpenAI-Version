class QuizQuestion:
    def __init__(self):
        self.prompts = {
            "quiz_question_prompt": """You are given a five-option multiple-choice quiz.

Task:
Select the option (A, B, C, D, or E) that is the exact original dataset instance.

Important definition of exact:
Exact means lexical identity, not semantic similarity.
Choose the option that preserves the original wording most faithfully.
Do not reward paraphrases, synonyms, style changes, or rewritten phrases.

Rules:
1. Output only one uppercase letter: A, B, C, D, or E.
2. Do not output any explanation, punctuation, or extra text.
3. If no option appears to be the exact original, output E.
4. Do not guess based on topic familiarity or world knowledge.
5. Base your decision only on wording-level comparison among options.

Decision policy:
- Compare options by wording, token choice, and phrase form.
- Prefer the option with minimal lexical deviation from the original form.
- If uncertain between multiple options, output E.

Options:
A)
{option_a}

B)
{option_b}

C)
{option_c}

D)
{option_d}

E)
None of the provided options.

Final answer (single letter only):
"""
        }

    def get_prompt(self, prompt_type):
        return self.prompts.get(prompt_type, "Invalid prompt type")
