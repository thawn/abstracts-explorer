"""
Evaluator facade composing generation and running mixins.
"""

from typing import Optional

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.evaluation.errors import EvaluationError
from abstracts_explorer.evaluation.generator import GeneratorMixin
from abstracts_explorer.evaluation.runner import RunnerMixin


class Evaluator(GeneratorMixin, RunnerMixin):
    """
    Automatic evaluation of the RAG system.

    Wraps all evaluation operations — Q/A pair generation, evaluation
    execution, and result storage — sharing a single
    :class:`~abstracts_explorer.embeddings.EmbeddingsManager` and its
    ``openai_client`` for LLM calls.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        A connected embeddings manager.  Its ``openai_client`` property is
        used for all LLM calls (generation and judging).
    db : DatabaseManager
        A connected database manager.
    model : str, optional
        Chat model name.  Falls back to config default.

    Examples
    --------
    >>> em = EmbeddingsManager()
    >>> em.create_collection()
    >>> with DatabaseManager() as db:
    ...     evaluator = Evaluator(em, db)
    ...     pairs = evaluator.generate_qa_pairs(n_pairs_per_tool=2)
    ...     evaluator.store_qa_pairs(pairs)
    ...     run_id = evaluator.run_evaluation()
    ...     print(evaluator.format_run_summary(run_id))
    """

    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        db: DatabaseManager,
        model: Optional[str] = None,
    ):
        if embeddings_manager is None:
            raise EvaluationError("embeddings_manager is required.")
        if db is None:
            raise EvaluationError("db is required.")

        config = get_config()
        self.embeddings_manager = embeddings_manager
        self.db = db
        self.model = model or config.chat_model
        self.request_timeout = config.llm_request_timeout

    @property
    def openai_client(self):
        """
        OpenAI client shared with the embeddings manager.

        Returns
        -------
        OpenAI
            The lazily-initialised client from ``EmbeddingsManager``.
        """
        return self.embeddings_manager.openai_client
