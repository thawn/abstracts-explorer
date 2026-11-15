"""
Evaluation Q/A pair and result repositories.
"""

import logging
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import select, delete, func

# Import Pydantic models from plugin framework

# Import SQLAlchemy models
from abstracts_explorer.db_models import (
    EvalQAPair,
    EvalResult,
)

from abstracts_explorer.db.errors import DatabaseError
from abstracts_explorer.db.session import SessionMixin

logger = logging.getLogger(__name__)


class EvaluationRepoMixin(SessionMixin):
    """Evaluation Q/A pair and result repositories."""

    # ------------------------------------------------------------------ #
    #  Evaluation Q/A pair and result methods                              #
    # ------------------------------------------------------------------ #

    def add_eval_qa_pair(
        self,
        conversation_id: str,
        turn_number: int,
        query: str,
        expected_answer: str,
        tool_name: Optional[str] = None,
        source_info: Optional[str] = None,
    ) -> int:
        """
        Insert a single evaluation Q/A pair.

        Parameters
        ----------
        conversation_id : str
            Identifier grouping turns in a conversation.
        turn_number : int
            Position within the conversation (0 = first).
        query : str
            The user query text.
        expected_answer : str
            The expected/reference answer.
        tool_name : str, optional
            MCP tool expected to be invoked.
        source_info : str, optional
            JSON metadata about how the pair was generated.

        Returns
        -------
        int
            Primary key of the inserted row.

        Raises
        ------
        DatabaseError
            If insertion fails.
        """
        try:
            pair = EvalQAPair(
                conversation_id=conversation_id,
                turn_number=turn_number,
                query=query,
                expected_answer=expected_answer,
                tool_name=tool_name,
                source_info=source_info,
            )
            self.session.add(pair)
            self.session.commit()
            return cast(int, pair.id)
        except Exception as e:
            raise self._db_error("Failed to add eval QA pair", e)

    def get_eval_qa_pairs(
        self,
        verified_only: bool = False,
        tool_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evaluation Q/A pairs with optional filters.

        Parameters
        ----------
        verified_only : bool
            If ``True``, return only pairs with ``verified == 1``.
        tool_name : str, optional
            Filter by expected MCP tool name.
        conversation_id : str, optional
            Filter by conversation.
        limit : int, optional
            Maximum number of pairs to return.
        offset : int
            Number of rows to skip (for pagination).

        Returns
        -------
        list of dict
            Matching Q/A pairs as dictionaries.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        try:
            stmt = select(EvalQAPair)
            if verified_only:
                stmt = stmt.where(EvalQAPair.verified == 1)
            if tool_name:
                stmt = stmt.where(EvalQAPair.tool_name == tool_name)
            if conversation_id:
                stmt = stmt.where(EvalQAPair.conversation_id == conversation_id)
            stmt = stmt.order_by(EvalQAPair.conversation_id, EvalQAPair.turn_number)
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            rows = self.session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "conversation_id": r.conversation_id,
                    "turn_number": r.turn_number,
                    "query": r.query,
                    "expected_answer": r.expected_answer,
                    "tool_name": r.tool_name,
                    "verified": r.verified,
                    "source_info": r.source_info,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        except Exception as e:
            raise DatabaseError(f"Failed to get eval QA pairs: {str(e)}") from e

    def get_eval_qa_pair_count(self, verified_only: bool = False) -> int:
        """
        Count evaluation Q/A pairs.

        Parameters
        ----------
        verified_only : bool
            If ``True``, count only verified pairs.

        Returns
        -------
        int
            Number of matching pairs.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        try:
            stmt = select(func.count()).select_from(EvalQAPair)
            if verified_only:
                stmt = stmt.where(EvalQAPair.verified == 1)
            return self.session.execute(stmt).scalar() or 0
        except Exception as e:
            raise DatabaseError(f"Failed to count eval QA pairs: {str(e)}") from e

    def update_eval_qa_pair(self, pair_id: int, **fields) -> bool:
        """
        Update fields on an existing Q/A pair.

        Parameters
        ----------
        pair_id : int
            Primary key of the pair to update.
        **fields
            Keyword arguments mapping column names to new values.
            Supported keys: ``query``, ``expected_answer``, ``tool_name``,
            ``verified``, ``source_info``.

        Returns
        -------
        bool
            ``True`` if a row was updated, ``False`` if the pair was not found.

        Raises
        ------
        DatabaseError
            If update fails.
        """
        allowed = {"query", "expected_answer", "tool_name", "verified", "source_info"}
        to_set = {k: v for k, v in fields.items() if k in allowed}
        if not to_set:
            return False

        try:
            pair = self.session.get(EvalQAPair, pair_id)
            if pair is None:
                return False
            for k, v in to_set.items():
                setattr(pair, k, v)
            self.session.commit()
            return True
        except Exception as e:
            raise self._db_error("Failed to update eval QA pair", e)

    def delete_verified_eval_qa_pairs(self) -> int:
        """
        Delete all verified (accepted) evaluation Q/A pairs.

        Returns
        -------
        int
            Number of pairs deleted.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        try:
            result = self.session.execute(delete(EvalQAPair).where(EvalQAPair.verified == 1))
            count = getattr(result, "rowcount", 0) or 0
            self.session.commit()
            return count
        except Exception as e:
            raise self._db_error("Failed to delete verified eval QA pairs", e)

    def delete_eval_results(self, run_id: Optional[str] = None) -> int:
        """
        Delete stored evaluation results, optionally filtered to a single run.

        Parameters
        ----------
        run_id : str, optional
            If supplied, only results for this run are deleted.
            If ``None``, **all** stored results are deleted.

        Returns
        -------
        int
            Number of rows deleted.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        try:
            stmt = delete(EvalResult)
            if run_id is not None:
                stmt = stmt.where(EvalResult.run_id == run_id)
            result = self.session.execute(stmt)
            count = getattr(result, "rowcount", 0) or 0
            self.session.commit()
            return count
        except Exception as e:
            raise self._db_error("Failed to delete eval results", e)

    def add_eval_result(
        self,
        run_id: str,
        qa_pair_id: int,
        actual_answer: Optional[str] = None,
        actual_tool_name: Optional[str] = None,
        answer_score: Optional[float] = None,
        tool_correct: Optional[int] = None,
        latency_ms: Optional[int] = None,
        error: Optional[str] = None,
        judge_reasoning: Optional[str] = None,
    ) -> int:
        """
        Insert a single evaluation result.

        Parameters
        ----------
        run_id : str
            Identifier for the evaluation run.
        qa_pair_id : int
            ID of the evaluated Q/A pair.
        actual_answer : str, optional
            Answer produced by the RAG system.
        actual_tool_name : str, optional
            MCP tool actually invoked.
        answer_score : float, optional
            LLM-judged quality score (1–5).
        tool_correct : int, optional
            1 if the correct tool was used, 0 otherwise.
        latency_ms : int, optional
            Query latency in milliseconds.
        error : str, optional
            Error message if the query failed.
        judge_reasoning : str, optional
            LLM judge's reasoning for the score.

        Returns
        -------
        int
            Primary key of the inserted row.

        Raises
        ------
        DatabaseError
            If insertion fails.
        """
        try:
            result = EvalResult(
                run_id=run_id,
                qa_pair_id=qa_pair_id,
                actual_answer=actual_answer,
                actual_tool_name=actual_tool_name,
                answer_score=answer_score,
                tool_correct=tool_correct,
                latency_ms=latency_ms,
                error=error,
                judge_reasoning=judge_reasoning,
            )
            self.session.add(result)
            self.session.commit()
            return cast(int, result.id)
        except Exception as e:
            raise self._db_error("Failed to add eval result", e)

    def get_eval_results(
        self,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evaluation results with optional run filter.

        Parameters
        ----------
        run_id : str, optional
            Filter by evaluation run. If ``None``, return results from all runs.
        limit : int, optional
            Maximum number of results to return.
        offset : int
            Number of rows to skip.

        Returns
        -------
        list of dict
            Evaluation results as dictionaries.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        try:
            stmt = select(EvalResult)
            if run_id:
                stmt = stmt.where(EvalResult.run_id == run_id)
            stmt = stmt.order_by(EvalResult.id)
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            rows = self.session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "run_id": r.run_id,
                    "qa_pair_id": r.qa_pair_id,
                    "actual_answer": r.actual_answer,
                    "actual_tool_name": r.actual_tool_name,
                    "answer_score": r.answer_score,
                    "tool_correct": r.tool_correct,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "judge_reasoning": r.judge_reasoning,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        except Exception as e:
            raise DatabaseError(f"Failed to get eval results: {str(e)}") from e

    def get_eval_run_ids(self) -> List[str]:
        """
        Return distinct evaluation run IDs ordered by run time, oldest first.

        The ordering is determined by the minimum ``created_at`` timestamp of
        all results in each run, so the most recent run appears last.

        Returns
        -------
        list of str
            Distinct run IDs ordered chronologically (oldest to newest).

        Raises
        ------
        DatabaseError
            If query fails.
        """
        try:
            stmt: Any = (
                select(EvalResult.run_id).group_by(EvalResult.run_id).order_by(func.min(EvalResult.created_at).asc())
            )
            return [row.run_id for row in self.session.execute(stmt).all()]
        except Exception as e:
            raise DatabaseError(f"Failed to get eval run IDs: {str(e)}") from e

    def get_eval_run_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Compute summary statistics for an evaluation run.

        Parameters
        ----------
        run_id : str
            The evaluation run identifier.

        Returns
        -------
        dict
            Dictionary with keys:

            - total : int – number of evaluated pairs
            - avg_score : float or None – mean answer quality score
            - tool_accuracy : float or None – fraction of correct tool selections
            - avg_latency_ms : float or None – mean latency
            - error_count : int – number of queries that produced errors
            - run_date : datetime or None – timestamp of the first result in the run

        Raises
        ------
        DatabaseError
            If query fails.
        """
        try:
            results = self.get_eval_results(run_id=run_id)
            if not results:
                return {
                    "total": 0,
                    "avg_score": None,
                    "tool_accuracy": None,
                    "avg_latency_ms": None,
                    "error_count": 0,
                    "run_date": None,
                }

            total = len(results)
            scores = [r["answer_score"] for r in results if r["answer_score"] is not None]
            tool_vals = [r["tool_correct"] for r in results if r["tool_correct"] is not None]
            latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
            errors = sum(1 for r in results if r["error"])

            # Determine the timestamp of the earliest result in this run
            stmt: Any = select(func.min(EvalResult.created_at)).where(EvalResult.run_id == run_id)
            run_date = self.session.execute(stmt).scalar()

            return {
                "total": total,
                "avg_score": (sum(scores) / len(scores)) if scores else None,
                "tool_accuracy": (sum(tool_vals) / len(tool_vals)) if tool_vals else None,
                "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
                "error_count": errors,
                "run_date": run_date,
            }
        except Exception as e:
            raise DatabaseError(f"Failed to compute eval run summary: {str(e)}") from e
