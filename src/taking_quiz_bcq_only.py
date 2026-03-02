import pandas as pd

from core.bias_compensator_quiz import BiasCompensatorQuiz
from helpers.logging_config import configure_logger
from helpers.option_counter import calculate_non_preferred_options
from services.taking_quiz_argparse_handler import TakingQuizArgumentParser

logger = configure_logger(__name__)


def main():
    args = TakingQuizArgumentParser().parse_args()
    df = pd.read_csv(args.filepath, encoding="utf-8")

    # Load existing BDQ results from the CSV
    if 'bdq_results' not in df.columns:
        raise ValueError("No 'bdq_results' column found. Run full taking_quiz.py first to generate BDQ results.")
    
    logger.info("Loading existing BDQ results from CSV...")
    non_preferred_options, bdq_results = calculate_non_preferred_options(df)
    logger.info(f"Non-preferred options: {', '.join(non_preferred_options).upper()}")

    # Run only BCQ
    BiasCompensatorQuiz(
        df=df,
        args=args,
        non_preferred_options=non_preferred_options,
        bdq_results=bdq_results,
    ).process()

    logger.info("*** BCQ process done! ***")


if __name__ == "__main__":
    main()
