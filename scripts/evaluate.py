"""Run the retrieval/faithfulness evaluation harness against a dataset file.

Usage:
    python -m scripts.evaluate data/eval/dataset.yaml
    python -m scripts.evaluate data/eval/dataset.yaml --top-k 3 --skip-judge

Why this exists: mirrors scripts/ingest.py -- the evaluation logic lives in
app/services/evaluation.py and is HTTP-agnostic, so this script only parses
arguments, wires the real services via the composition root, and prints the
report. No orchestration logic belongs here.
"""

import argparse
import logging
import sys
from pathlib import Path

from app.api.deps import get_evaluation_service
from app.core.config import get_settings
from app.core.exceptions import WerbyError
from app.core.logging import configure_logging
from app.services.evaluation import format_report, load_dataset

logger = logging.getLogger("scripts.evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Werby's retrieval hit-rate and answer faithfulness."
    )
    parser.add_argument("dataset", type=Path, help="Path to a YAML eval dataset")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override the configured retrieval_top_k",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the LLM faithfulness judge (avoids the extra model call and cost)",
    )
    args = parser.parse_args()

    configure_logging(get_settings().log_level)

    try:
        cases = load_dataset(args.dataset)
    except WerbyError as exc:
        logger.error("Failed to load dataset: %s", exc.message)
        return 1

    service = get_evaluation_service()
    try:
        report = service.run(cases, top_k=args.top_k, use_judge=not args.skip_judge)
    except WerbyError as exc:
        logger.error(
            "Evaluation failed: %s. If the corpus is empty, run "
            "'python -m scripts.ingest' first.",
            exc.message,
        )
        return 1

    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
