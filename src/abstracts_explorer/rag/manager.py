"""
RAG chat facade.

:class:`RAGChat` composes the agent, conversation, and extraction mixins.
"""

import logging

from abstracts_explorer.rag.agent import AgentMixin
from abstracts_explorer.rag.conversation import ConversationMixin
from abstracts_explorer.rag.extraction import ExtractionMixin

logger = logging.getLogger(__name__)


class RAGChat(AgentMixin, ConversationMixin, ExtractionMixin):
    """
    Retrieval-Augmented Generation chat for conference abstracts.

    The agent uses MCP tools to search papers, analyze topics, and understand
    trends, then generates contextual responses. See :class:`~abstracts_explorer.rag.agent.AgentMixin`
    for constructor parameters and :meth:`query` for usage.
    """
