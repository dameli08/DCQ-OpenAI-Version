class OptionGeneration:
    def __init__(self):
        self.prompts = {
            "option_generation_prompt": """Your task is to create a four-choice quiz by replacing the words in the provided INPUT TEXT with their contextually relevant synonyms.
The meaning and sentence structure of the four options MUST EXACTLY match every detail in the INPUT TEXT.
You MUST NOT include the provided INPUT TEXT as an option.

You MUST make sure that:
(1) You generate EXACTLY FOUR COMPLETE DISTINCT options A, B, C, D based on the provided INPUT TEXT;
(2) The ONLY difference between options is WORD-LEVEL PERTURBATIONS;
(3) Options are ORDERED as A, B, C, D;
(4) There is NOT any extra explanation, no markdown, no headings, no extra text;
(5) You follow the EXACT FORMAT below with NO VARIATIONS;
(6) You comply with every specific symbol and letter detail in the given INPUT TEXT;
(7) All options retain the EXACT LABEL from the INPUT TEXT, if there is one;
(8) GENERATE ALL FOUR OPTIONS COMPLETELY - DO NOT TRUNCATE OR SHORTEN;
(9) Each option must be fully complete and not cut off mid-sentence.

CRITICAL REQUIREMENTS:
- Return ALL FOUR options A, B, C, D completely
- Do not skip any option
- Do not truncate any option
- Each option must be a complete, well-formed alternative

---
INPUT TEXT:
{original_instance}
---
FORMAT (STRICTLY FOLLOW THIS EXACTLY):

A)
First word-level perturbation goes here completely

B)
Second word-level perturbation goes here completely

C)
Third word-level perturbation goes here completely

D)
Fourth word-level perturbation goes here completely

---
IMPORTANT: Generate ALL FOUR options. Every option (A, B, C, D) must be complete. All four are required without exception.
"""
        }

    def get_prompt(self, prompt_type):
        return self.prompts.get(prompt_type, "Invalid prompt type")
