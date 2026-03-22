import math
import re

from helpers.logging_config import configure_logger

logger = configure_logger(__name__)


def letter_counter(df, column_name, letter):
    """Count occurrences of a letter (A-E) in BDQ/BCQ results.
    Handles various response formats: 'A', 'a', 'A)', 'a)', 'A)\n', etc."""
    # Extract just the first character and convert to lowercase
    # This handles responses like 'E', 'E)', 'E\n', etc.
    def extract_letter(val):
        if pd.isna(val):
            return ""
        val_str = str(val).strip()
        if not val_str:
            return ""
        # Get first character and lowercase it
        return val_str[0].lower()
    
    import pandas as pd
    first_letters = df[column_name].apply(extract_letter)
    total_count = (first_letters == letter.lower()).sum()
    return total_count


def calculate_non_preferred_options(df, bdq_results_column_name="bdq_results"):
    bdq_results = {letter: 0 for letter in "abcde"}
    random_chance_count = math.ceil(len(df) / 5)

    for letter in bdq_results:
        bdq_results[letter] = letter_counter(df, bdq_results_column_name, letter)

    non_preferred_options = [
        letter
        for letter, count in bdq_results.items()
        if count < random_chance_count and letter != "e"
    ]

    if not non_preferred_options:
        non_preferred_options = ["a", "b", "c", "d"]

    return non_preferred_options, bdq_results


class QuizPerformance:
    def __init__(self, df, non_preferred_options, bdq_results):
        self.df = df
        self.non_preferred_options = non_preferred_options
        self.bdq_results = bdq_results
        self.bcqs_results = {}

    def calculate_best_quiz_performance(self):
        self._calculate_quiz_performance()

        triples = [
            (letter, self.bcqs_results[letter][letter], self.bdq_results[letter])
            for letter in self.bcqs_results
        ]

        if not triples:
            raise ValueError("No BCQ results found to estimate contamination.")

        triples.sort(key=lambda x: (-x[1], x[2]))

        max_count = triples[0][1]
        max_cont_positional_bias = triples[0][2]
        max_cont_level = max_count / len(self.df)

        kappa = (max_count - max_cont_positional_bias) / (
            len(self.df) - max_cont_positional_bias
        )

        # Edge case: only one non-preferred option was selected, so there is no
        # second BCQ run to compare against.
        if len(triples) == 1:
            min_cont_level = kappa
        else:
            second_max_count = triples[1][1]
            min_cont_level = max(kappa, second_max_count / len(self.df))

        return max_cont_level, min_cont_level, self.bcqs_results

    def _calculate_quiz_performance(self):
        for non_preferred_option in self.non_preferred_options:
            temp_results = {letter: 0 for letter in "abcde"}

            for letter in temp_results:
                temp_results[letter] = letter_counter(
                    df=self.df,
                    column_name=f"bcq_results_for_position_{non_preferred_option}",
                    letter=letter,
                )
            self.bcqs_results[non_preferred_option] = temp_results
