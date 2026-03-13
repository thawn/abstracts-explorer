"""
Automatic Evaluation
====================

This module implements automatic evaluation of the RAG system.  It provides
functions for:

* Generating evaluation Q/A pairs from the paper database and MCP tools
  using an LLM.
* Running the RAG pipeline on stored Q/A pairs and scoring the output
  with an LLM-as-judge approach.
* Computing summary statistics and formatting results for display.
"""

import json
import logging
import random
import time
import uuid
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import get_config
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Exception raised for evaluation-related errors."""

    pass


# ---------------------------------------------------------------------------
#  Prompt templates
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTIONS: Dict[str, str] = {
    "search_papers": ("Search for papers on a specific topic. Returns relevant papers from the database."),
    "get_cluster_topics": ("Get the main research topics from clustering analysis of paper embeddings."),
    "analyze_topic_relevance": (
        "Analyze the relevance and popularity of a research topic by counting "
        "papers within a specified distance in embedding space."
    ),
    "get_topic_evolution": ("Analyze how a specific topic has evolved over the years."),
    "get_cluster_visualization": ("Generate visualization data for clustered paper embeddings."),
}

_QA_GENERATION_SYSTEM_PROMPT = """\
You are an expert evaluation dataset creator for a conference paper search and \
analysis system.  The system allows users to search papers, analyze research \
topics, track topic evolution, and visualize clusters.

Your task is to generate realistic evaluation query/answer pairs that test the \
system's capabilities.  Each pair consists of a user *query* and the *expected \
answer* the system should produce.

IMPORTANT RULES:
1. Queries must be natural questions a researcher would ask.
2. Answers should be based ONLY on the provided paper information.
3. Each query must clearly map to one of the available MCP tools.
4. Include specific details from the papers (titles, topics, years) in answers.
5. Keep answers concise but informative (2-5 sentences).
"""

_QA_GENERATION_USER_PROMPT = """\
Generate {n_pairs} evaluation query/answer pair(s) for the MCP tool \
"{tool_name}".

Tool description: {tool_description}

Here are some papers from the database to base your queries on:
{papers_context}

Respond with a JSON array.  Each element must have these keys:
- "query": the user's natural-language question
- "expected_answer": a concise reference answer based on the papers above

Example format:
[
  {{
    "query": "What are the main topics in NeurIPS 2025?",
    "expected_answer": "Based on the analysis ..."
  }}
]

Return ONLY the JSON array, no other text.
"""

_FOLLOWUP_GENERATION_PROMPT = """\
Given the following initial query and answer, generate {n_followups} natural \
follow-up question(s) and their expected answers.  The follow-ups should \
explore the topic deeper or ask for clarification.

Initial query: {initial_query}
Initial answer: {initial_answer}

Available papers for context:
{papers_context}

Respond with a JSON array.  Each element must have these keys:
- "query": the follow-up question
- "expected_answer": the expected answer

Return ONLY the JSON array, no other text.
"""

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating the quality of an AI assistant's answer \
about conference papers.  Compare the actual answer to the expected reference \
answer.

Score the answer on a scale of 1-5:
  5 = Excellent: fully correct, comprehensive, well-structured
  4 = Good: mostly correct with minor omissions
  3 = Adequate: partially correct but missing important details
  2 = Poor: significant inaccuracies or missing key information
  1 = Bad: incorrect, irrelevant, or empty response

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}
"""

_JUDGE_USER_PROMPT = """\
Query: {query}

Expected answer:
{expected_answer}

Actual answer:
{actual_answer}

Score the actual answer (1-5) and explain briefly.
"""


# ---------------------------------------------------------------------------
#  Q/A pair generation
# ---------------------------------------------------------------------------


def _get_openai_client(
    lm_studio_url: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> OpenAI:
    """
    Create an OpenAI client from config or explicit parameters.

    Parameters
    ----------
    lm_studio_url : str, optional
        API base URL.  Falls back to config.
    auth_token : str, optional
        Bearer token.  Falls back to config.

    Returns
    -------
    OpenAI
        Initialised client.
    """
    config = get_config()
    url = (lm_studio_url or config.llm_backend_url).rstrip("/")
    token = auth_token if auth_token is not None else config.llm_backend_auth_token
    return OpenAI(base_url=f"{url}/v1", api_key=token or "lm-studio-local")


def _sample_papers_context(db: DatabaseManager, n_papers: int = 10) -> str:
    """
    Sample random papers from the database and format them as context.

    Parameters
    ----------
    db : DatabaseManager
        Connected database manager.
    n_papers : int
        Number of papers to sample.

    Returns
    -------
    str
        Formatted paper context string.
    """
    total = db.get_paper_count()
    if total == 0:
        return "(no papers in database)"

    # Fetch a random page of papers
    all_papers = db.search_papers(limit=min(total, 200))
    sampled = random.sample(all_papers, min(n_papers, len(all_papers)))

    lines: List[str] = []
    for i, p in enumerate(sampled, 1):
        title = p.get("title", "Untitled")
        abstract = (p.get("abstract") or "")[:200]
        year = p.get("year", "")
        conf = p.get("conference", "")
        keywords = p.get("keywords", "")
        lines.append(f"Paper {i}: {title} ({conf} {year})\n" f"  Keywords: {keywords}\n" f"  Abstract: {abstract}...")
    return "\n\n".join(lines)


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
    import re

    # Strip <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Failed to parse LLM JSON output: {exc}\nRaw output:\n{text[:500]}") from exc

    if not isinstance(parsed, list):
        raise EvaluationError(f"Expected JSON array, got {type(parsed).__name__}")
    return parsed


def generate_qa_pairs(
    db: DatabaseManager,
    n_pairs_per_tool: int = 2,
    tools: Optional[List[str]] = None,
    generate_followups: bool = True,
    n_followups: int = 1,
    model: Optional[str] = None,
    lm_studio_url: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate evaluation Q/A pairs using an LLM.

    For each requested MCP tool a set of query/answer pairs is generated
    based on papers sampled from the database.  Optionally, follow-up
    questions are generated for each initial pair.

    Parameters
    ----------
    db : DatabaseManager
        Connected database with papers.
    n_pairs_per_tool : int
        Number of initial Q/A pairs to generate per tool.
    tools : list of str, optional
        MCP tool names to generate pairs for.  Defaults to all tools.
    generate_followups : bool
        Whether to generate follow-up questions.
    n_followups : int
        Number of follow-up turns per initial pair.
    model : str, optional
        Chat model name.  Falls back to config default.
    lm_studio_url : str, optional
        LLM backend URL.  Falls back to config default.
    auth_token : str, optional
        Auth token.  Falls back to config default.

    Returns
    -------
    list of dict
        Generated pairs, each with keys: ``conversation_id``,
        ``turn_number``, ``query``, ``expected_answer``, ``tool_name``,
        ``source_info``.

    Raises
    ------
    EvaluationError
        If generation fails.
    """
    config = get_config()
    client = _get_openai_client(lm_studio_url, auth_token)
    chat_model = model or config.chat_model

    available_tools = list(_TOOL_DESCRIPTIONS.keys())
    target_tools = tools if tools else available_tools
    invalid = set(target_tools) - set(available_tools)
    if invalid:
        raise EvaluationError(f"Unknown tool(s): {', '.join(sorted(invalid))}")

    papers_context = _sample_papers_context(db)
    all_pairs: List[Dict[str, Any]] = []

    for tool_name in target_tools:
        logger.info(f"Generating {n_pairs_per_tool} Q/A pair(s) for tool: {tool_name}")
        tool_desc = _TOOL_DESCRIPTIONS[tool_name]

        user_prompt = _QA_GENERATION_USER_PROMPT.format(
            n_pairs=n_pairs_per_tool,
            tool_name=tool_name,
            tool_description=tool_desc,
            papers_context=papers_context,
        )

        try:
            resp = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": _QA_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=120,
            )
            raw = resp.choices[0].message.content or ""
            pairs = _parse_json_array(raw)
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"LLM call failed for tool {tool_name}: {exc}") from exc

        for pair in pairs[:n_pairs_per_tool]:
            conv_id = uuid.uuid4().hex[:12]
            entry = {
                "conversation_id": conv_id,
                "turn_number": 0,
                "query": pair.get("query", ""),
                "expected_answer": pair.get("expected_answer", ""),
                "tool_name": tool_name,
                "source_info": json.dumps({"model": chat_model, "tool": tool_name}),
            }
            all_pairs.append(entry)

            # Generate follow-ups
            if generate_followups and n_followups > 0:
                followup_prompt = _FOLLOWUP_GENERATION_PROMPT.format(
                    n_followups=n_followups,
                    initial_query=entry["query"],
                    initial_answer=entry["expected_answer"],
                    papers_context=papers_context,
                )
                try:
                    fu_resp = client.chat.completions.create(
                        model=chat_model,
                        messages=[
                            {"role": "system", "content": _QA_GENERATION_SYSTEM_PROMPT},
                            {"role": "user", "content": followup_prompt},
                        ],
                        temperature=0.7,
                        max_tokens=2000,
                        timeout=120,
                    )
                    fu_raw = fu_resp.choices[0].message.content or ""
                    followups = _parse_json_array(fu_raw)
                except Exception as exc:
                    logger.warning(f"Failed to generate follow-ups for conv {conv_id}: {exc}")
                    followups = []

                for idx, fu in enumerate(followups[:n_followups], 1):
                    all_pairs.append(
                        {
                            "conversation_id": conv_id,
                            "turn_number": idx,
                            "query": fu.get("query", ""),
                            "expected_answer": fu.get("expected_answer", ""),
                            "tool_name": tool_name,
                            "source_info": json.dumps(
                                {"model": chat_model, "tool": tool_name, "followup_of": conv_id}
                            ),
                        }
                    )

    logger.info(f"Generated {len(all_pairs)} Q/A pairs total")
    return all_pairs


def store_qa_pairs(db: DatabaseManager, pairs: List[Dict[str, Any]]) -> int:
    """
    Persist generated Q/A pairs into the database.

    Parameters
    ----------
    db : DatabaseManager
        Connected database.
    pairs : list of dict
        Pairs as returned by :func:`generate_qa_pairs`.

    Returns
    -------
    int
        Number of pairs stored.
    """
    count = 0
    for p in pairs:
        db.add_eval_qa_pair(
            conversation_id=p["conversation_id"],
            turn_number=p["turn_number"],
            query=p["query"],
            expected_answer=p["expected_answer"],
            tool_name=p.get("tool_name"),
            source_info=p.get("source_info"),
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
#  Evaluation runner
# ---------------------------------------------------------------------------


def _judge_answer(
    client: OpenAI,
    model: str,
    query: str,
    expected_answer: str,
    actual_answer: str,
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to score an answer.

    Parameters
    ----------
    client : OpenAI
        OpenAI client.
    model : str
        Judge model name.
    query : str
        Original query.
    expected_answer : str
        Reference answer.
    actual_answer : str
        RAG system output.

    Returns
    -------
    dict
        ``{"score": int, "reasoning": str}``
    """
    import re

    user_prompt = _JUDGE_USER_PROMPT.format(
        query=query,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
            timeout=60,
        )
        raw = resp.choices[0].message.content or ""
        # Strip <think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Strip markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned.strip())
        score = max(1, min(5, int(parsed.get("score", 3))))
        return {"score": score, "reasoning": parsed.get("reasoning", "")}
    except Exception as exc:
        logger.warning(f"Judge scoring failed: {exc}")
        return {"score": None, "reasoning": f"Judge error: {exc}"}


def run_evaluation(
    db: DatabaseManager,
    embeddings_manager: Any,
    model: Optional[str] = None,
    lm_studio_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    verified_only: bool = True,
    limit: Optional[int] = None,
) -> str:
    """
    Run evaluation on stored Q/A pairs and record results.

    Executes each stored query through the RAG system, scores the output
    with an LLM judge, and stores the results in the database.

    Parameters
    ----------
    db : DatabaseManager
        Connected database containing Q/A pairs.
    embeddings_manager : EmbeddingsManager
        Connected embeddings manager for RAG.
    model : str, optional
        Chat model to use.  Falls back to config.
    lm_studio_url : str, optional
        LLM backend URL.  Falls back to config.
    auth_token : str, optional
        Auth token.  Falls back to config.
    verified_only : bool
        If ``True``, only evaluate verified pairs (default).
    limit : int, optional
        Maximum number of pairs to evaluate.

    Returns
    -------
    str
        The ``run_id`` for the evaluation run.

    Raises
    ------
    EvaluationError
        If evaluation fails.
    """
    from .rag import RAGChat

    config = get_config()
    chat_model = model or config.chat_model
    client = _get_openai_client(lm_studio_url, auth_token)

    pairs = db.get_eval_qa_pairs(verified_only=verified_only, limit=limit)
    if not pairs:
        raise EvaluationError("No Q/A pairs found for evaluation. Generate and verify pairs first.")

    run_id = f"eval-{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting evaluation run {run_id} with {len(pairs)} pair(s)")

    # Group pairs by conversation for multi-turn handling
    conversations: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        conv_id = p["conversation_id"]
        conversations.setdefault(conv_id, []).append(p)
    for turns in conversations.values():
        turns.sort(key=lambda x: x["turn_number"])

    # Initialise a RAGChat instance for each conversation
    for conv_id, turns in conversations.items():
        rag = RAGChat(
            embeddings_manager=embeddings_manager,
            database=db,
            lm_studio_url=lm_studio_url,
            model=chat_model,
        )

        for pair in turns:
            qa_pair_id = pair["id"]
            query_text = pair["query"]
            expected = pair["expected_answer"]
            expected_tool = pair.get("tool_name")

            start = time.time()
            actual_answer = None
            actual_tool = None
            error_msg = None

            try:
                result = rag.query(query_text)
                actual_answer = result.get("response", "")
                tools_used = result.get("metadata", {}).get("tools_executed", [])
                actual_tool = tools_used[0] if tools_used else None
            except Exception as exc:
                error_msg = str(exc)
                logger.warning(f"Query failed for pair {qa_pair_id}: {exc}")

            elapsed_ms = int((time.time() - start) * 1000)

            # Tool correctness
            tool_correct = None
            if expected_tool and actual_tool is not None:
                tool_correct = 1 if actual_tool == expected_tool else 0

            # LLM judge scoring
            judge_result: Dict[str, Any] = {"score": None, "reasoning": ""}
            if actual_answer and not error_msg:
                judge_result = _judge_answer(client, chat_model, query_text, expected, actual_answer)

            db.add_eval_result(
                run_id=run_id,
                qa_pair_id=qa_pair_id,
                actual_answer=actual_answer,
                actual_tool_name=actual_tool,
                answer_score=judge_result.get("score"),
                tool_correct=tool_correct,
                latency_ms=elapsed_ms,
                error=error_msg,
                judge_reasoning=judge_result.get("reasoning", ""),
            )

    logger.info(f"Evaluation run {run_id} complete")
    return run_id


# ---------------------------------------------------------------------------
#  Results formatting
# ---------------------------------------------------------------------------


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
        f"Total pairs evaluated: {summary['total']}",
    ]

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
        lines.append(f"Expected: {qa_pair['expected_answer'][:200]}...")
    else:
        lines.append(f"QA Pair ID: {result['qa_pair_id']}")

    actual = result.get("actual_answer") or "(no answer)"
    lines.append(f"Actual:   {actual[:200]}...")

    score = result.get("answer_score")
    lines.append(f"Score:    {score}/5" if score else "Score:    N/A")

    tc = result.get("tool_correct")
    if tc is not None:
        expected_tool = qa_pair.get("tool_name", "?") if qa_pair else "?"
        lines.append(
            f"Tool:     {'✅ correct' if tc else '❌ wrong'} (expected: {expected_tool}, got: {result.get('actual_tool_name', '?')})"
        )

    if result.get("latency_ms"):
        lines.append(f"Latency:  {result['latency_ms']} ms")

    if result.get("error"):
        lines.append(f"Error:    {result['error']}")

    reasoning = result.get("judge_reasoning")
    if reasoning:
        lines.append(f"Judge:    {reasoning[:200]}")

    return "\n".join(lines)
