"""
OpenAI-compatible client setup and model compatibility checks.
"""

import logging
import sys
from typing import Any, Optional

import httpx2
from openai import OpenAI

from abstracts_explorer.database import DatabaseManager, normalize_model_name
from abstracts_explorer.embeddings.errors import translate_errors
from abstracts_explorer.rate_limiter import RateLimitedTransport
from abstracts_explorer.utils.http_utils import build_llm_timeout

logger = logging.getLogger(__name__)


class ModelMixin:
    """Lazy OpenAI client creation, connection testing, and model checks."""

    _openai_client: Optional[OpenAI]
    _init_lock: Any
    backend_url: str
    llm_backend_auth_token: Optional[str]
    model_name: str
    requests_per_minute: int
    request_timeout: float
    connect_timeout: float

    @property
    def openai_client(self) -> OpenAI:
        """
        Get the OpenAI client, creating it lazily on first access.

        The client always uses an explicit structured timeout. When
        ``requests_per_minute`` is greater than 0 a
        :class:`RateLimitedTransport` is wrapped around the default httpx2
        transport so that every HTTP request is automatically throttled.

        Returns
        -------
        OpenAI
            Initialized OpenAI client instance.
        """
        if self._openai_client is None:
            with self._init_lock:
                if self._openai_client is None:
                    self._openai_client = self._create_openai_client()
        return self._openai_client

    def _create_openai_client(self) -> OpenAI:
        """Create the OpenAI client with an explicit structured timeout."""
        transport: Optional[Any] = None
        if self.requests_per_minute > 0:
            transport = RateLimitedTransport(httpx2.HTTPTransport(), self.requests_per_minute)
        http_client = httpx2.Client(
            transport=transport,
            timeout=build_llm_timeout(self.request_timeout, self.connect_timeout),
        )
        return OpenAI(
            base_url=f"{self.backend_url}/v1",
            api_key=self.llm_backend_auth_token or "lm-studio-local",
            http_client=http_client,
        )

    def test_api_connection(
        self,
        *,
        verbose: bool = False,
        model_hint: Optional[str] = None,
    ) -> bool:
        """
        Test the connection to the OpenAI-compatible API endpoint.

        Parameters
        ----------
        verbose : bool, optional
            If True, print user-facing status messages, by default False.
        model_hint : str, optional
            Description of the expected model shown in the verbose failure
            message. If None, a message using ``self.model_name`` is shown.

        Returns
        -------
        bool
            True if the connection is successful, False otherwise.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> if em.test_api_connection():
        ...     print("API is accessible")
        """
        try:
            _ = self.openai_client.models.list()
            logger.debug("Successfully connected to OpenAI-compatible API at %s", self.backend_url)
            if verbose:
                print("✅ Successfully connected to OpenAI API")
            return True
        except Exception as e:
            logger.warning("Failed to connect to OpenAI-compatible API: %s", str(e))
            if verbose:
                print("\n❌ Failed to connect to OpenAI API!", file=sys.stderr)
                print("\nPlease ensure:", file=sys.stderr)
                print(f"  - OpenAI-compatible API is running at {self.backend_url}", file=sys.stderr)
                hint = model_hint or f"The {self.model_name} model is loaded"
                print(f"  - {hint}", file=sys.stderr)
            return False

    def check_model_compatibility(self) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Check if the current embedding model matches the one stored in the database.

        Returns
        -------
        tuple of (bool, str or None, str or None)
            - compatible: True if models match or no model is stored, False if they differ
            - stored_model: Name of the model stored in the database, or None if not set
            - current_model: Name of the current model

        Raises
        ------
        EmbeddingsError
            If database operations fail.

        Examples
        --------
        >>> em = EmbeddingsManager()
        >>> compatible, stored, current = em.check_model_compatibility()
        >>> if not compatible:
        ...     print(f"Model mismatch: stored={stored}, current={current}")
        """
        with translate_errors("Failed to check model compatibility"):
            with DatabaseManager() as db_manager:
                stored_model = db_manager.get_embedding_model()

            if stored_model is None:
                return True, None, self.model_name
            return (
                normalize_model_name(stored_model) == normalize_model_name(self.model_name),
                stored_model,
                self.model_name,
            )
