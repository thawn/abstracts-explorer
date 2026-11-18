"""
Embedding generation with token budgeting and context-overflow retries.
"""

import logging
from collections.abc import Sequence
from typing import Any, Optional

from abstracts_explorer.embeddings.errors import EmbeddingsError
from abstracts_explorer.embeddings.tokens import (
    _CONTEXT_RETRY_SHRINK,
    _MAX_CONTEXT_RETRIES,
    context_length_limit_from_error,
    count_tokens,
    is_context_length_error,
    token_margin,
    truncate_with_counts,
)

logger = logging.getLogger(__name__)


class GenerationMixin:
    """Generate embeddings for single texts and batches."""

    max_input_tokens: int
    _learned_token_budget: Optional[int]
    model_name: str
    openai_client: Any

    def _effective_token_budget(self) -> int:
        """
        Return the token budget used before sending text to the API.

        The configured model limit is reduced by a safety margin to account for
        tokenizer differences. A smaller budget learned from a previous API
        context-length error takes precedence when proactive truncation is
        enabled. A non-positive ``max_input_tokens`` disables truncation
        entirely and also ignores any learned budget.

        Returns
        -------
        int
            Token budget, or 0 when proactive truncation is disabled.
        """
        limit = self.max_input_tokens
        if limit <= 0:
            return 0
        budget = max(1, limit - token_margin(limit))

        if self._learned_token_budget is not None:
            budget = min(budget, self._learned_token_budget)

        return budget

    def _prepare_embedding_input(self, text: str) -> str:
        """
        Truncate *text* to the effective token budget.

        Parameters
        ----------
        text : str
            Text that will be sent to the embedding API.

        Returns
        -------
        str
            Original text when it fits, otherwise a truncated prefix.
        """
        budget = self._effective_token_budget()
        if budget <= 0:
            return text

        truncated, original_tokens, truncated_tokens = truncate_with_counts(text, budget)
        if truncated == text:
            return text

        if original_tokens is not None:
            logger.warning(
                "Truncated embedding input from %d to %d tokens (budget %d) for model %s",
                original_tokens,
                truncated_tokens,
                budget,
                self.model_name,
            )
        else:
            logger.warning(
                "Truncated embedding input from %d to %d characters for model %s",
                len(text),
                len(truncated),
                self.model_name,
            )
        return truncated

    def _shrink_embedding_input(self, original_text: str, current_text: str, error: Exception) -> str:
        """
        Produce a shorter input after the backend rejects a context-length overflow.

        Parameters
        ----------
        original_text : str
            Full, untruncated text.
        current_text : str
            Input that the backend just rejected.
        error : Exception
            Error raised by the embedding API.

        Returns
        -------
        str
            Shortened input to retry.
        """
        current_tokens = count_tokens(current_text)
        if current_tokens is None:
            return current_text[: max(1, len(current_text) // 2)]

        target = max(1, int(current_tokens * _CONTEXT_RETRY_SHRINK))
        parsed_limit = context_length_limit_from_error(error)
        if parsed_limit is not None:
            target = min(target, max(1, parsed_limit - token_margin(parsed_limit)))
        target = min(target, max(1, current_tokens - 1))

        self._learned_token_budget = target
        truncated, _, _ = truncate_with_counts(original_text, target)
        return truncated

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a given text using OpenAI-compatible API.

        Rate limiting (if configured via ``requests_per_minute``) is handled
        transparently by the underlying ``httpx2`` transport.

        Inputs longer than the model's context window are truncated before the
        request. If the backend still reports an overflow, the input is shrunk
        and retried a few times.

        Parameters
        ----------
        text : str
            Text to generate embedding for.

        Returns
        -------
        list of float
            Embedding vector.

        Raises
        ------
        EmbeddingsError
            If embedding generation fails.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> embedding = em.generate_embedding("Sample text")
        >>> len(embedding)
        4096
        """
        if not text or not text.strip():
            raise EmbeddingsError("Cannot generate embedding for empty text")

        prepared_text = self._prepare_embedding_input(text)

        attempt = 0
        while True:
            try:
                response = self.openai_client.embeddings.create(model=self.model_name, input=prepared_text)
                if not response.data:
                    raise EmbeddingsError("No embedding data in API response")

                embedding = response.data[0].embedding
                logger.debug("Generated embedding with dimension: %d", len(embedding))
                return embedding
            except EmbeddingsError:
                raise
            except Exception as e:
                if not is_context_length_error(e) or attempt >= _MAX_CONTEXT_RETRIES:
                    raise EmbeddingsError(f"Failed to generate embedding via OpenAI API: {str(e)}") from e

                prepared_text = self._shrink_embedding_input(text, prepared_text, e)
                attempt += 1
                logger.warning(
                    "Embedding input exceeded the model context window; retrying with a shorter input "
                    "(attempt %d/%d)",
                    attempt,
                    _MAX_CONTEXT_RETRIES,
                )

    def generate_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for several texts in a single API request.

        Texts are truncated to the effective token budget like
        :meth:`generate_embedding`. If the backend rejects the batch because of
        the context window (or returns an unexpected number of vectors), each
        text is retried individually with :meth:`generate_embedding`.

        Parameters
        ----------
        texts : sequence of str
            Texts to generate embeddings for.

        Returns
        -------
        list of list of float
            Embedding vectors in the same order as *texts*.

        Raises
        ------
        EmbeddingsError
            If a text is empty or embedding generation fails.
        """
        if not texts:
            return []

        texts = list(texts)
        for text in texts:
            if not text or not text.strip():
                raise EmbeddingsError("Cannot generate embedding for empty text")

        prepared = [self._prepare_embedding_input(text) for text in texts]

        fallback_reason: Optional[str] = None
        try:
            response = self.openai_client.embeddings.create(model=self.model_name, input=prepared)
            if len(response.data) != len(prepared):
                fallback_reason = "unexpected number of embeddings in API response"
            else:
                indices = [item.index for item in response.data]
                if sorted(indices) != list(range(len(prepared))):
                    fallback_reason = "invalid embedding indices in API response"
                else:
                    ordered = sorted(response.data, key=lambda item: item.index)
                    return [item.embedding for item in ordered]
        except Exception as e:
            if is_context_length_error(e):
                fallback_reason = "input exceeded the model context window"
            else:
                fallback_reason = f"batch request failed: {e}"

        logger.warning("Batch embedding failed (%s); retrying texts individually", fallback_reason)
        return [self.generate_embedding(text) for text in texts]
