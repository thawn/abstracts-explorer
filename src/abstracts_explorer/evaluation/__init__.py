"""
Automatic RAG evaluation package.

The public API is re-exported here so ``from abstracts_explorer.evaluation
import ...`` keeps working.
"""

from abstracts_explorer.evaluation.errors import EvaluationError
from abstracts_explorer.evaluation.formatting import (
    _parse_json_array,
    format_eval_result_detail,
    format_eval_summary,
)
from abstracts_explorer.evaluation.manager import Evaluator

__all__ = [
    "Evaluator",
    "EvaluationError",
    "_parse_json_array",
    "format_eval_summary",
    "format_eval_result_detail",
]
