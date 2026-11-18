"""
Result formatting and JSON cleanup for evaluation.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from abstracts_explorer.evaluation.errors import EvaluationError
from abstracts_explorer.utils.time_utils import format_timestamp

logger = logging.getLogger(__name__)


def _strip_reasoning_and_fences(text: str) -> str:
    """
    Remove hidden-reasoning blocks and markdown code fences from an LLM
    response.

    Reasoning models may emit hidden-reasoning blocks before the answer, and
    answers are often wrapped in markdown code fences.  Both are removed so
    the remaining text can be parsed as JSON.

    Parameters
    ----------
    text : str
        Raw LLM response.

    Returns
    -------
    str
        Cleaned text, stripped of surrounding whitespace.
    """
    # Strip hidden-reasoning blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Parse a JSON array from LLM output, tolerating markdown fences.

    Parameters
    ----------
    text : str
        Raw LLM output that should contain a JSON array.

    Returns
    -------
    list of dict
        Parsed array elements.

    Raises
    ------
    EvaluationError
        If parsing fails.
    """
    cleaned = _strip_reasoning_and_fences(text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Failed to parse LLM JSON output: {exc}\nRaw output:\n{text[:500]}") from exc

    if not isinstance(parsed, list):
        raise EvaluationError(f"Expected JSON array, got {type(parsed).__name__}")
    return parsed


def format_eval_summary(summary: Dict[str, Any], run_id: str) -> str:
    """
    Format an evaluation run summary for display.

    Parameters
    ----------
    summary : dict
        Summary from :meth:`DatabaseManager.get_eval_run_summary`.
    run_id : str
        The evaluation run identifier.

    Returns
    -------
    str
        Human-readable summary string.
    """
    lines = [
        f"Evaluation Run: {run_id}",
        "=" * 50,
    ]

    run_date = summary.get("run_date")
    if run_date is not None:
        date_str = format_timestamp(run_date, "%Y-%m-%d %H:%M:%S", fallback=str(run_date))
        lines.append(f"Run date:              {date_str}")

    lines.append(f"Total pairs evaluated: {summary['total']}")

    avg = summary.get("avg_score")
    if avg is not None:
        lines.append(f"Average answer score:  {avg:.2f} / 5.00")
    else:
        lines.append("Average answer score:  N/A")

    acc = summary.get("tool_accuracy")
    if acc is not None:
        lines.append(f"Tool selection accuracy: {acc * 100:.1f}%")
    else:
        lines.append("Tool selection accuracy: N/A")

    lat = summary.get("avg_latency_ms")
    if lat is not None:
        lines.append(f"Average latency:       {lat:.0f} ms")
    else:
        lines.append("Average latency:       N/A")

    lines.append(f"Errors:                {summary.get('error_count', 0)}")
    return "\n".join(lines)


def format_eval_result_detail(result: Dict[str, Any], qa_pair: Optional[Dict[str, Any]] = None) -> str:
    """
    Format a single evaluation result for display.

    Parameters
    ----------
    result : dict
        Result row from :meth:`DatabaseManager.get_eval_results`.
    qa_pair : dict, optional
        Corresponding Q/A pair for additional context.

    Returns
    -------
    str
        Human-readable detail string.
    """
    lines = [f"Result #{result['id']}  (run: {result['run_id']})"]
    lines.append("-" * 50)

    if qa_pair:
        lines.append(f"Query:    {qa_pair['query']}")
        lines.append(f"Expected: {qa_pair['expected_answer']}")
    else:
        lines.append(f"QA Pair ID: {result['qa_pair_id']}")

    actual = result.get("actual_answer") or "(no answer)"
    lines.append(f"Actual:   {actual}")

    score = result.get("answer_score")
    lines.append(f"Score:    {score}/5" if score else "Score:    N/A")

    tc = result.get("tool_correct")
    if tc is not None:
        expected_tool = qa_pair.get("tool_name", "?") if qa_pair else "?"
        lines.append(
            f"Tool:     {'✅ correct' if tc else '❌ wrong'}"
            f" (expected: {expected_tool}, got: {result.get('actual_tool_name', '?')})"
        )

    if result.get("latency_ms"):
        lines.append(f"Latency:  {result['latency_ms']} ms")

    if result.get("error"):
        lines.append(f"Error:    {result['error']}")

    reasoning = result.get("judge_reasoning")
    if reasoning:
        lines.append(f"Judge:    {reasoning}")

    return "\n".join(lines)
