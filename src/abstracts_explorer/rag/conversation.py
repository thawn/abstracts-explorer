"""
RAG conversation state and export.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pydantic_ai.messages import ModelRequest

logger = logging.getLogger(__name__)


class ConversationMixin:
    """Reset, inspect, and export the conversation."""

    _message_history: List[Any]

    def _update_message_history_instructions(self, instructions: str) -> None:
        """
        Update the instructions on the first ModelRequest in message history.

        Pydantic AI does not re-add system prompts on continuation turns.
        When the user changes the conference in the UI, the instructions need to
        reflect the new conference, but the first ModelRequest in
        ``_message_history`` still holds the stale system prompt from the prior
        conference.  This method walks the history and updates the ``instructions``
        field on every ModelRequest so the LLM sees consistent context.
        """
        for msg in self._message_history:
            if isinstance(msg, ModelRequest):
                msg.instructions = instructions

    def reset_conversation(self):
        """
        Reset conversation history.

        Examples
        --------
        >>> chat.reset_conversation()
        """
        self._message_history = []
        logger.info("Conversation history reset")

    @property
    def conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history as a list of role/content dicts.

        This property converts Pydantic AI message objects to simple
        dicts for backwards compatibility with code that accesses
        ``conversation_history`` directly.

        Returns
        -------
        list of dict
            Messages in ``{"role": "user"|"assistant", "content": "..."}`` format.
        """
        history: List[Dict[str, str]] = []
        for msg in self._message_history:
            kind = getattr(msg, "kind", None)
            if kind == "request":
                for part in msg.parts:
                    part_kind = getattr(part, "part_kind", None)
                    if part_kind == "user-prompt":
                        content = getattr(part, "content", None)
                        if isinstance(content, str):
                            history.append({"role": "user", "content": content})
            elif kind == "response":
                for part in msg.parts:
                    part_kind = getattr(part, "part_kind", None)
                    if part_kind == "text":
                        content = getattr(part, "content", None)
                        if isinstance(content, str):
                            history.append({"role": "assistant", "content": content})
        return history

    def export_conversation(self, output_path: Path) -> None:
        """
        Export conversation history to JSON file.

        Parameters
        ----------
        output_path : Path
            Path to output JSON file.

        Examples
        --------
        >>> chat.export_conversation("conversation.json")
        """
        output_path = Path(output_path)
        history = self.conversation_history
        with open(output_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Conversation exported to {output_path}")
