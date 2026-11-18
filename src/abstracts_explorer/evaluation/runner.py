"""
Evaluation runner and LLM-as-judge scoring.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from abstracts_explorer.evaluation.errors import EvaluationError
from abstracts_explorer.evaluation.formatting import _strip_reasoning_and_fences, format_eval_summary
from abstracts_explorer.evaluation.prompts import _JUDGE_SYSTEM_PROMPT, _JUDGE_USER_PROMPT
from abstracts_explorer.llm_utils import chat_completion_text

logger = logging.getLogger(__name__)


class RunnerMixin:
    """Run stored Q/A pairs through RAG and score answers."""

    db: Any
    model: str
    openai_client: Any
    embeddings_manager: Any

    def _judge_answer(
        self,
        query: str,
        expected_answer: str,
        actual_answer: str,
    ) -> Dict[str, Any]:
        """
        Use LLM-as-judge to score an answer.

        Parameters
        ----------
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
        user_prompt = _JUDGE_USER_PROMPT.format(
            query=query,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
        )
        try:
            raw = chat_completion_text(
                client=self.openai_client,
                model=self.model,
                system_prompt=_JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=300,
                timeout=60,
            )
            logger.debug(f"Judge raw response: {raw[:200]}")
            cleaned = _strip_reasoning_and_fences(raw)
            # Use raw_decode to extract the first JSON object even if the model
            # adds preamble text (e.g. "Here is my evaluation: {…}")
            decoder = json.JSONDecoder()
            start = cleaned.find("{")
            if start == -1:
                raise ValueError(f"No JSON object found in judge response: {cleaned[:200]!r}")
            parsed, _ = decoder.raw_decode(cleaned, start)
            score_raw = parsed.get("score")
            if score_raw is None:
                logger.warning("Judge returned null score, defaulting to 3")
                score = 3
            else:
                score = max(1, min(5, int(score_raw)))
            return {"score": score, "reasoning": parsed.get("reasoning", "")}
        except Exception as exc:
            logger.warning(f"Judge scoring failed: {exc}")
            return {"score": None, "reasoning": f"Judge error: {exc}"}

    def run_evaluation(
        self,
        verified_only: bool = True,
        limit: Optional[int] = None,
    ) -> str:
        """
        Run evaluation on stored Q/A pairs and record results.

        Executes each stored query through the RAG system, scores the
        output with an LLM judge, and stores the results in the database.

        Parameters
        ----------
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
        from abstracts_explorer.rag import RAGChat

        pairs = self.db.get_eval_qa_pairs(verified_only=verified_only, limit=limit)
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
                embeddings_manager=self.embeddings_manager,
                database=self.db,
                model=self.model,
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

                conferences = None
                years = None
                source_info = pair.get("source_info")
                if source_info:
                    try:
                        parsed_source = json.loads(source_info) if isinstance(source_info, str) else source_info
                        tool_args = parsed_source.get("tool_args", {}) if isinstance(parsed_source, dict) else {}
                        if isinstance(tool_args, dict):
                            raw_conferences = tool_args.get("conferences") or tool_args.get("conference")
                            if isinstance(raw_conferences, str):
                                conferences = [raw_conferences]
                            elif isinstance(raw_conferences, list):
                                conferences = [str(item) for item in raw_conferences]
                            raw_years = tool_args.get("years") or tool_args.get("year")
                            if isinstance(raw_years, int):
                                years = [raw_years]
                            elif isinstance(raw_years, list):
                                years = [int(item) for item in raw_years if str(item).isdigit()]
                    except (TypeError, ValueError, json.JSONDecodeError):
                        logger.debug("Could not parse source_info context for pair %s", qa_pair_id)

                try:
                    result = rag.query(query_text, conferences=conferences, years=years)
                    actual_answer = (result.get("response") or "").strip()
                    tools_used = result.get("metadata", {}).get("tools_executed", [])
                    actual_tool = tools_used[0] if tools_used else None
                except Exception as exc:
                    error_msg = str(exc)
                    logger.warning(f"Query failed for pair {qa_pair_id}: {exc}")

                elapsed_ms = int((time.time() - start) * 1000)

                # Tool correctness
                tool_correct = None
                if expected_tool and actual_tool is not None:
                    tool_correct = 1 if expected_tool in tools_used else 0

                # LLM judge scoring
                judge_result: Dict[str, Any] = {"score": None, "reasoning": ""}
                if actual_answer and not error_msg:
                    judge_result = self._judge_answer(query_text, expected, actual_answer)
                elif not error_msg:
                    logger.warning(f"Skipping judge for pair {qa_pair_id}: empty response from RAG")

                self.db.add_eval_result(
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

    def format_run_summary(self, run_id: str) -> str:
        """
        Compute and format the summary for *run_id*.

        Parameters
        ----------
        run_id : str
            Evaluation run identifier.

        Returns
        -------
        str
            Human-readable summary.
        """
        summary = self.db.get_eval_run_summary(run_id)
        return format_eval_summary(summary, run_id)
