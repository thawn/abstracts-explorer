"""
Q/A pair generation from the paper database and MCP tools.
"""

import json
import logging
import random
import uuid
from typing import Any, Dict, List, Optional

from abstracts_explorer.evaluation.errors import EvaluationError
from abstracts_explorer.evaluation.formatting import _parse_json_array
from abstracts_explorer.evaluation.prompts import (
    _FOLLOWUP_GENERATION_PROMPT,
    _QA_GENERATION_SYSTEM_PROMPT,
    _QA_GENERATION_USER_PROMPT,
)
from abstracts_explorer.llm_utils import chat_completion_text
from abstracts_explorer.mcp_tools import (
    TOOL_ARGUMENT_EXAMPLES,
    TOOL_DESCRIPTIONS,
    execute_mcp_tool,
    format_tool_result_for_llm,
)

logger = logging.getLogger(__name__)


class GeneratorMixin:
    """Generate and store evaluation Q/A pairs."""

    db: Any
    model: str
    openai_client: Any
    embeddings_manager: Any

    def _sample_papers_context(self, n_papers: int = 10) -> str:
        """
        Sample random papers from the database and format them as context.

        Parameters
        ----------
        n_papers : int
            Number of papers to sample.

        Returns
        -------
        str
            Formatted paper context string.
        """
        total = self.db.get_paper_count()
        if total == 0:
            return "(no papers in database)"

        # Fetch a random page of papers
        all_papers = self.db.search_papers(limit=min(total, 200))
        sampled = random.sample(all_papers, min(n_papers, len(all_papers)))

        lines: List[str] = []
        for i, p in enumerate(sampled, 1):
            title = p.get("title", "Untitled")
            abstract = (p.get("abstract") or "")[:200]
            year = p.get("year", "")
            conf = p.get("conference", "")
            keywords = p.get("keywords", "")
            lines.append(f"Paper {i}: {title} ({conf} {year})\n  Keywords: {keywords}\n  Abstract: {abstract}...")
        return "\n\n".join(lines)

    def generate_qa_pairs(
        self,
        n_pairs_per_tool: int = 2,
        tools: Optional[List[str]] = None,
        generate_followups: bool = True,
        n_followups: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate evaluation Q/A pairs using the LLM.

        For each requested MCP tool a set of query/answer pairs is
        generated based on papers sampled from the database.  Optionally,
        follow-up questions are generated for each initial pair.

        Parameters
        ----------
        n_pairs_per_tool : int
            Number of initial Q/A pairs to generate per tool.
        tools : list of str, optional
            MCP tool names to generate pairs for.  Defaults to all tools.
        generate_followups : bool
            Whether to generate follow-up questions.
        n_followups : int
            Number of follow-up turns per initial pair.

        Returns
        -------
        list of dict
            Generated pairs, each with keys: ``conversation_id``,
            ``turn_number``, ``query``, ``expected_answer``,
            ``tool_name``, ``source_info``.

        Raises
        ------
        EvaluationError
            If generation fails.
        """
        available_tools = list(TOOL_DESCRIPTIONS.keys())
        target_tools = tools if tools else available_tools
        invalid = set(target_tools) - set(available_tools)
        if invalid:
            raise EvaluationError(f"Unknown tool(s): {', '.join(sorted(invalid))}")

        papers_context = self._sample_papers_context()
        all_pairs: List[Dict[str, Any]] = []

        for tool_name in target_tools:
            logger.info(f"Generating {n_pairs_per_tool} Q/A pair(s) for tool: {tool_name}")
            tool_desc = TOOL_DESCRIPTIONS[tool_name]
            tool_args_example = TOOL_ARGUMENT_EXAMPLES[tool_name]

            user_prompt = _QA_GENERATION_USER_PROMPT.format(
                n_pairs=n_pairs_per_tool,
                tool_name=tool_name,
                tool_description=tool_desc,
                tool_arguments_example=tool_args_example,
                papers_context=papers_context,
            )

            try:
                raw = chat_completion_text(
                    client=self.openai_client,
                    model=self.model,
                    system_prompt=_QA_GENERATION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    max_tokens=2000,
                    timeout=120,
                )
                pairs = _parse_json_array(raw)
            except EvaluationError:
                raise
            except Exception as exc:
                raise EvaluationError(f"LLM call failed for tool {tool_name}: {exc}") from exc

            for pair in pairs[:n_pairs_per_tool]:
                conv_id = uuid.uuid4().hex[:12]
                query = pair.get("query", "")
                tool_args = pair.get("tool_arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                # Execute the MCP tool to obtain the real expected answer
                expected_answer = self._run_tool_for_answer(tool_name, tool_args, conv_id, turn=0)
                if expected_answer is None:
                    continue  # tool returned an error; skip this pair

                entry = {
                    "conversation_id": conv_id,
                    "turn_number": 0,
                    "query": query,
                    "expected_answer": expected_answer,
                    "tool_name": tool_name,
                    "source_info": json.dumps({"model": self.model, "tool": tool_name, "tool_args": tool_args}),
                }
                all_pairs.append(entry)
                logger.info(
                    f"[{tool_name}] turn 0 (conv {conv_id})\n"
                    f"  Query:    {entry['query']}\n"
                    f"  Expected: {entry['expected_answer']}"
                )

                # Generate follow-ups
                if generate_followups and n_followups > 0:
                    followup_prompt = _FOLLOWUP_GENERATION_PROMPT.format(
                        n_followups=n_followups,
                        initial_query=entry["query"],
                        initial_answer=entry["expected_answer"],
                        tool_name=tool_name,
                        tool_arguments_example=tool_args_example,
                        papers_context=papers_context,
                    )
                    try:
                        fu_raw = chat_completion_text(
                            client=self.openai_client,
                            model=self.model,
                            system_prompt=_QA_GENERATION_SYSTEM_PROMPT,
                            user_prompt=followup_prompt,
                            temperature=0.7,
                            max_tokens=2000,
                            timeout=120,
                        )
                        followups = _parse_json_array(fu_raw)
                    except Exception as exc:
                        logger.warning(f"Failed to generate follow-ups for conv {conv_id}: {exc}")
                        followups = []

                    for idx, fu in enumerate(followups[:n_followups], 1):
                        fu_query = fu.get("query", "")
                        fu_tool_args = fu.get("tool_arguments", {})
                        if isinstance(fu_tool_args, str):
                            try:
                                fu_tool_args = json.loads(fu_tool_args)
                            except json.JSONDecodeError:
                                fu_tool_args = {}

                        fu_answer = self._run_tool_for_answer(tool_name, fu_tool_args, conv_id, turn=idx)
                        if fu_answer is None:
                            continue  # tool returned an error; skip this follow-up

                        fu_entry = {
                            "conversation_id": conv_id,
                            "turn_number": idx,
                            "query": fu_query,
                            "expected_answer": fu_answer,
                            "tool_name": tool_name,
                            "source_info": json.dumps(
                                {
                                    "model": self.model,
                                    "tool": tool_name,
                                    "followup_of": conv_id,
                                    "tool_args": fu_tool_args,
                                }
                            ),
                        }
                        all_pairs.append(fu_entry)
                        logger.info(
                            f"[{tool_name}] turn {idx} (conv {conv_id})\n"
                            f"  Query:    {fu_entry['query']}\n"
                            f"  Expected: {fu_entry['expected_answer']}"
                        )

        logger.info(f"Generated {len(all_pairs)} Q/A pairs total")
        return all_pairs

    def _run_tool_for_answer(
        self, tool_name: str, tool_args: Dict[str, Any], conv_id: str, turn: int
    ) -> Optional[str]:
        """
        Execute an MCP tool and return its formatted output as an expected answer.

        Parameters
        ----------
        tool_name : str
            The MCP tool to execute.
        tool_args : Dict[str, Any]
            Arguments for the tool call.
        conv_id : str
            Conversation ID (used only for log messages).
        turn : int
            Turn number (used only for log messages).

        Returns
        -------
        str or None
            Formatted tool output, or ``None`` if the tool returned an error or
            the result is not a valid JSON string.
        """
        try:
            raw_result = execute_mcp_tool(tool_name, tool_args)
            if not isinstance(raw_result, str):
                logger.warning(
                    f"MCP tool {tool_name} returned unexpected type {type(raw_result).__name__} "
                    f"(conv {conv_id}, turn {turn}) — skipping pair"
                )
                return None
            try:
                result_data = json.loads(raw_result)
            except json.JSONDecodeError as exc:
                logger.warning(
                    f"MCP tool {tool_name} returned invalid JSON (conv {conv_id}, turn {turn}): {exc} — skipping pair"
                )
                return None
            if isinstance(result_data, dict) and "error" in result_data:
                logger.warning(
                    f"MCP tool {tool_name} returned error (conv {conv_id}, turn {turn}): "
                    f"{result_data['error']} — skipping pair"
                )
                return None
            return format_tool_result_for_llm(tool_name, raw_result)
        except Exception as exc:
            logger.warning(f"MCP tool {tool_name} failed (conv {conv_id}, turn {turn}): {exc} — skipping pair")
            return None

    def store_qa_pairs(self, pairs: List[Dict[str, Any]]) -> int:
        """
        Persist generated Q/A pairs into the database.

        Parameters
        ----------
        pairs : list of dict
            Pairs as returned by :meth:`generate_qa_pairs`.

        Returns
        -------
        int
            Number of pairs stored.
        """
        count = 0
        for p in pairs:
            self.db.add_eval_qa_pair(
                conversation_id=p["conversation_id"],
                turn_number=p["turn_number"],
                query=p["query"],
                expected_answer=p["expected_answer"],
                tool_name=p.get("tool_name"),
                source_info=p.get("source_info"),
            )
            count += 1
        return count
