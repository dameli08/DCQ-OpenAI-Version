class QuizQuestion:
    def __init__(self):
        self.prompts = {
            "quiz_question_prompt": """You are provided with a quiz task from the {dataset_name} dataset ({split_name} split).
Your ONLY job is to identify which option (A, B, C, D, or E) is the EXACT ORIGINAL INSTANCE.

RULES:
1. Return ONLY a single letter (A, B, C, D, or E) - nothing else
2. Options A, B, C, D are word-level variations of the original with some words replaced by synonyms
3. Find the option that is IDENTICAL to the original instance - the one with NO word replacements
4. If NONE of the options match the original exactly, return E

ANALYSIS METHOD:
- Compare each option word-by-word with all others
- The original option will have slightly different word choices compared to variations
- Look for the option that appears most "different" from the others in specific words
- The other 3 options will share many words in common (because they are variations)
- The option that stands OUT as having unique word choices is likely the original

---
OPTION A:
{option_a}

OPTION B:
{option_b}

OPTION C:
{option_c}

OPTION D:
{option_d}

OPTION E:
None of these match the original.

---
Which option is the EXACT ORIGINAL (A, B, C, D, or E)? Answer with a single letter:
"""
        }

    def get_prompt(self, prompt_type):
        return self.prompts.get(prompt_type, "Invalid prompt type")
